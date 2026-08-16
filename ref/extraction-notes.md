<task id="ses_012e6effbffeJjKD50TOijuyBD" state="completed">
<task_result>
I have everything needed. Here is the complete extraction, organized by function, verbatim with line numbers.

---

# FILE 1: `/home/shinde/.local/share/opencode/tool-output/tool_fed18c9fc001qs0ktlg6o5wzyp` — llama.cpp `src/llama-graph.cpp` (3780 lines)

## 1.1 `llm_graph_context::build_norm` (lines 1556–1589)

```cpp
1556: ggml_tensor * llm_graph_context::build_norm(
1557:          ggml_tensor * cur,
1558:          ggml_tensor * mw,
1559:          ggml_tensor * mb,
1560:        llm_norm_type   type,
1561:                  int   il) const {
1562:     switch (type) {
1563:         case LLM_NORM:       cur = ggml_norm    (ctx0, cur, hparams.f_norm_eps);     break;
1564:         case LLM_NORM_RMS:   cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps); break;
1565:         case LLM_NORM_GROUP:
1566:             {
1567:                 cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, cur->ne[1]);
1568:                 cur = ggml_group_norm(ctx0, cur, hparams.n_norm_groups, hparams.f_norm_group_eps);
1569:                 cur = ggml_reshape_2d(ctx0, cur, cur->ne[0],    cur->ne[2]);
1570:             } break;
1571:     }
1572: 
1573:     if (mw || mb) {
1574:         cb(cur, "norm", il);
1575:     }
1576: 
1577:     if (mw) {
1578:         cur = ggml_mul(ctx0, cur, mw);
1579:         if (mb) {
1580:             cb(cur, "norm_w", il);
1581:         }
1582:     }
1583: 
1584:     if (mb) {
1585:         cur = ggml_add(ctx0, cur, mb);
1586:     }
1587: 
1588:     return cur;
1589: }
```

## 1.2 LoRA matmul helpers used by build_ffn / build_moe_ffn

`build_lora_mm` (lines 1487–1516):
```cpp
1487: ggml_tensor * llm_graph_context::build_lora_mm(
1488:           ggml_tensor * w,
1489:           ggml_tensor * cur,
1490:           ggml_tensor * w_s) const {
1491:     ggml_tensor * res = ggml_mul_mat(ctx0, w, cur);
1492: 
1493:     if (w_s) {
1494:         res = ggml_mul(ctx0, res, w_s);
1495:     }
1496: 
1497:     for (const auto & lora : *loras) {
1498:         llama_adapter_lora_weight * lw = lora.first->get_weight(w);
1499:         if (lw == nullptr) {
1500:             continue;
1501:         }
1502: 
1503:         const float adapter_scale = lora.second;
1504:         const float scale = lw->get_scale(lora.first->alpha, adapter_scale);
1505: 
1506:         ggml_tensor * ab_cur = ggml_mul_mat(
1507:                 ctx0, lw->b,
1508:                 ggml_mul_mat(ctx0, lw->a, cur)
1509:                 );
1510: 
1511:         ab_cur = ggml_scale(ctx0, ab_cur, scale);
1512:         res = ggml_add(ctx0, res, ab_cur);
1513:     }
1514: 
1515:     return res;
1516: }
```

`build_lora_mm_id` (lines 1518–1554):
```cpp
1518: ggml_tensor * llm_graph_context::build_lora_mm_id(
1519:           ggml_tensor * w,   // ggml_tensor * as
1520:           ggml_tensor * cur, // ggml_tensor * b
1521:           ggml_tensor * ids,
1522:           ggml_tensor * w_s) const {
1523:     ggml_tensor * res = ggml_mul_mat_id(ctx0, w, cur, ids);
1524: 
1525:     if (w_s) {
1526:         const int64_t n_expert = w_s->ne[0];
1527:         const int64_t n_tokens = cur->ne[2];
1528:         ggml_tensor * s = ggml_reshape_3d(ctx0, w_s, 1, n_expert, 1);
1529:         s = ggml_repeat_4d(ctx0, s, 1, n_expert, n_tokens, 1);
1530:         s = ggml_get_rows(ctx0, s, ids);
1531:         res = ggml_mul(ctx0, res, s);
1532:     }
1533:     for (const auto & lora : *loras) {
1534:         llama_adapter_lora_weight * lw = lora.first->get_weight(w);
1535:         if (lw == nullptr) {
1536:             continue;
1537:         }
1538: 
1539:         const float alpha = lora.first->alpha;
1540:         const float rank  = (float) lw->b->ne[0];
1541:         const float scale = alpha ? lora.second * alpha / rank : lora.second;
1542: 
1543:         ggml_tensor * ab_cur = ggml_mul_mat_id(
1544:                 ctx0, lw->b,
1545:                 ggml_mul_mat_id(ctx0, lw->a, cur, ids),
1546:                 ids
1547:                 );
1548: 
1549:         ab_cur = ggml_scale(ctx0, ab_cur, scale);
1550:         res = ggml_add(ctx0, res, ab_cur);
1551:     }
1552: 
1553:     return res;
1554: }
```

## 1.3 `llm_graph_context::build_ffn` (non-MoE FFN builder, lines 1669–1869)

```cpp
1669: ggml_tensor * llm_graph_context::build_ffn(
1670:          ggml_tensor * cur,
1671:          ggml_tensor * up,
1672:          ggml_tensor * up_b,
1673:          ggml_tensor * up_s,
1674:          ggml_tensor * gate,
1675:          ggml_tensor * gate_b,
1676:          ggml_tensor * gate_s,
1677:          ggml_tensor * down,
1678:          ggml_tensor * down_b,
1679:          ggml_tensor * down_s,
1680:          ggml_tensor * act_scales,
1681:      llm_ffn_op_type   type_op,
1682:    llm_ffn_gate_type   type_gate,
1683:                  int   il) const {
1684:     // NVFP4 support is currently restricted to
1685:     // 1) LORA absence (*_s would be applied after LORA residual, which is incorrect)
1686:     // 2) bias absense (*_s would be applied after bias addition, which is incorrect)
1687:     // TODO: disambiguate LLM-architectural scales (which use *_s) from NVFP4 scale_2 (which also uses *_s currently)
1688:     auto has_lora = [this](ggml_tensor * w) {
1689:         if (!w) {
1690:             return false;
1691:         }
1692:         for (const auto & lora : *loras) {
1693:             if (lora.first->get_weight(w) != nullptr) {
1694:                 return true;
1695:             }
1696:         }
1697:         return false;
1698:     };
1699: 
1700:     GGML_ASSERT(!up_s   || !up_b   || !up   || up->type   != GGML_TYPE_NVFP4);
1701:     GGML_ASSERT(!gate_s || !gate_b || !gate || gate->type != GGML_TYPE_NVFP4);
1702:     GGML_ASSERT(!down_s || !down_b || !down || down->type != GGML_TYPE_NVFP4);
1703:     GGML_ASSERT(!up_s   || !up   || up->type   != GGML_TYPE_NVFP4 || !has_lora(up));
1704:     GGML_ASSERT(!gate_s || !gate || gate->type != GGML_TYPE_NVFP4 || !has_lora(gate));
1705:     GGML_ASSERT(!down_s || !down || down->type != GGML_TYPE_NVFP4 || !has_lora(down));
1706: 
1707:     ggml_tensor * tmp = up ? build_lora_mm(up, cur) : cur;
1708:     cb(tmp, "ffn_up", il);
1709: 
1710:     if (up_b) {
1711:         tmp = ggml_add(ctx0, tmp, up_b);
1712:         cb(tmp, "ffn_up_b", il);
1713:     }
1714: 
1715:     if (up_s) {
1716:         tmp = ggml_mul(ctx0, tmp, up_s);
1717:         cb(tmp, "ffn_up_s", il);
1718:     }
1719: 
1720:     if (gate) {
1721:         switch (type_gate) {
1722:             case LLM_FFN_SEQ:
1723:                 {
1724:                     cur = build_lora_mm(gate, tmp);
1725:                     cb(cur, "ffn_gate", il);
1726:                 } break;
1727:             case LLM_FFN_PAR:
1728:                 {
1729:                     cur = build_lora_mm(gate, cur);
1730:                     cb(cur, "ffn_gate", il);
1731:                 } break;
1732:         }
1733: 
1734:         if (gate_b) {
1735:             cur = ggml_add(ctx0, cur, gate_b);
1736:             cb(cur, "ffn_gate_b", il);
1737:         }
1738: 
1739:         if (gate_s) {
1740:             cur = ggml_mul(ctx0, cur, gate_s);
1741:             cb(cur, "ffn_gate_s", il);
1742:         }
1743: 
1744:     } else {
1745:         cur = tmp;
1746:     }
1747: 
1748:     switch (type_op) {
1749:         case LLM_FFN_SILU:
1750:             if (gate && type_gate == LLM_FFN_PAR) {
1751:                 if (il >= 0) {
1752:                     const float limit = hparams.swiglu_clamp_shexp[il];
1753:                     constexpr float eps = 1e-6f;
1754:                     if (limit > eps) {
1755:                         tmp = ggml_clamp(ctx0, tmp, -limit, limit);
1756:                         cb(tmp, "ffn_up_clamped", il);
1757: 
1758:                         if (arch == LLM_ARCH_DEEPSEEK4 || (arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0)) {
1759:                             cur = ggml_clamp(ctx0, cur, -INFINITY, limit);
1760:                             cb(cur, "ffn_gate_clamped", il);
1761:                             cur = ggml_swiglu_split(ctx0, cur, tmp);
1762:                         } else {
1763:                             ggml_tensor * gate_act = ggml_silu(ctx0, cur);
1764:                             cb(gate_act, "ffn_silu", il);
1765:                             gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
1766:                             cb(gate_act, "ffn_silu_clamped", il);
1767:                             cur = ggml_mul(ctx0, gate_act, tmp);
1768:                         }
1769:                         cb(cur, "ffn_swiglu_limited", il);
1770:                         type_gate = LLM_FFN_SEQ;
1771:                         break;
1772:                     }
1773:                 }
1774: 
1775:                 cur = ggml_swiglu_split(ctx0, cur, tmp);
1776:                 cb(cur, "ffn_swiglu", il);
1777:                 type_gate = LLM_FFN_SEQ;
1778:             } else {
1779:                 cur = ggml_silu(ctx0, cur);
1780:                 cb(cur, "ffn_silu", il);
1781:             } break;
1782:         case LLM_FFN_GELU:
1783:             if (gate && type_gate == LLM_FFN_PAR) {
1784:                 cur = ggml_geglu_split(ctx0, cur, tmp);
1785:                 cb(cur, "ffn_geglu", il);
1786:                 type_gate = LLM_FFN_SEQ;
1787:             } else {
1788:                 cur = ggml_gelu(ctx0, cur);
1789:                 cb(cur, "ffn_gelu", il);
1790:                 if (act_scales != NULL) {
1791:                     cur = ggml_div(ctx0, cur, act_scales);
1792:                     cb(cur, "ffn_act", il);
1793:                 }
1794:             } break;
1795:         case LLM_FFN_RELU:
1796:             if (gate && type_gate == LLM_FFN_PAR) {
1797:                 cur = ggml_reglu_split(ctx0, cur, tmp);
1798:                 cb(cur, "ffn_reglu", il);
1799:                 type_gate = LLM_FFN_SEQ;
1800:             } else {
1801:                 cur = ggml_relu(ctx0, cur);
1802:                 cb(cur, "ffn_relu", il);
1803:             } break;
1804:         case LLM_FFN_RELU_SQR:
1805:             {
1806:                 cur = ggml_relu(ctx0, cur);
1807:                 cb(cur, "ffn_relu", il);
1808: 
1809:                 cur = ggml_sqr(ctx0, cur);
1810:                 cb(cur, "ffn_sqr(relu)", il);
1811:             } break;
1812:         case LLM_FFN_SWIGLU:
1813:             {
1814:                 cur = ggml_swiglu(ctx0, cur);
1815:                 cb(cur, "ffn_swiglu", il);
1816:             } break;
1817:         case LLM_FFN_SWIGLU_OAI_MOE:
1818:             if (gate && type_gate == LLM_FFN_PAR) {
1819:                 // same alpha/limit constants as gpt-oss
1820:                 const float alpha = 1.702f;
1821:                 const float limit = 7.0f;
1822:                 cur = ggml_swiglu_oai(ctx0, cur, tmp, alpha, limit);
1823:                 cb(cur, "ffn_swiglu_oai", il);
1824:                 type_gate = LLM_FFN_SEQ;
1825:             } else {
1826:                 GGML_ABORT("LLM_FFN_SWIGLU_OAI_MOE requires a parallel gate");
1827:             } break;
1828:         case LLM_FFN_GEGLU:
1829:             {
1830:                 cur = ggml_geglu(ctx0, cur);
1831:                 cb(cur, "ffn_geglu", il);
1832:             } break;
1833:         case LLM_FFN_REGLU:
1834:             {
1835:                 cur = ggml_reglu(ctx0, cur);
1836:                 cb(cur, "ffn_reglu", il);
1837:             } break;
1838:         default:
1839:             GGML_ABORT("fatal error");
1840:     }
1841: 
1842:     if (gate && type_gate == LLM_FFN_PAR) {
1843:         cur = ggml_mul(ctx0, cur, tmp);
1844:         cb(cur, "ffn_gate_par", il);
1845:     }
1846: 
1847:     if (down) {
1848:         cur = build_lora_mm(down, cur);
1849:         if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
1850:             // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
1851:             ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
1852:         }
1853:     }
1854: 
1855:     if (down_b) {
1856:         cb(cur, "ffn_down", il);
1857:     }
1858: 
1859:     if (down_b) {
1860:         cur = ggml_add(ctx0, cur, down_b);
1861:     }
1862: 
1863:     if (down_s) {
1864:         cur = ggml_mul(ctx0, cur, down_s);
1865:         cb(cur, "ffn_down_s", il);
1866:     }
1867: 
1868:     return cur;
1869: }
```

**Key semantics of `build_ffn`:** With `type_gate == LLM_FFN_PAR` (parallel gate, used by Qwen archs), the `up` projection is applied to the input first (`tmp`), then `gate` is applied to the ORIGINAL `cur` (not `tmp`), and finally for SILU: `cur = silu(gate) * up` (via `ggml_swiglu_split(cur, tmp)`), then `down` matmul. With `LLM_FFN_SEQ`, `gate` is applied sequentially on `tmp`. `act_scales` is only used by the GELU path.

## 1.4 `llm_graph_context::build_moe_ffn` — overload A (delegating, lines 1871–1913)

```cpp
1871: ggml_tensor * llm_graph_context::build_moe_ffn(
1872:          ggml_tensor * cur,
1873:          ggml_tensor * gate_inp,
1874:          ggml_tensor * up_exps,
1875:          ggml_tensor * gate_exps,
1876:          ggml_tensor * down_exps,
1877:          ggml_tensor * exp_probs_b,
1878:              int64_t   n_expert,
1879:              int64_t   n_expert_used,
1880:      llm_ffn_op_type   type_op,
1881:                 bool   norm_w,
1882:                float   w_scale,
1883:          llama_expert_gating_func_type gating_op,
1884:                  int   il,
1885:          ggml_tensor * probs_in,
1886:          ggml_tensor * gate_up_exps,
1887:          ggml_tensor * up_exps_s,
1888:          ggml_tensor * gate_exps_s,
1889:          ggml_tensor * down_exps_s,
1890:          ggml_tensor * selected_experts_in) const {
1891:     return build_moe_ffn(
1892:         cur,
1893:         gate_inp,  /* gate_inp_b  */ nullptr,
1894:         up_exps,   /* up_exps_b   */ nullptr,
1895:         gate_exps, /* gate_exps_b */ nullptr,
1896:         down_exps, /* down_exps_b */ nullptr,
1897:         exp_probs_b,
1898:         n_expert,
1899:         n_expert_used,
1900:         type_op,
1901:         norm_w,
1902:         w_scale,
1903:         gating_op,
1904:         il,
1905:         probs_in,
1906:         gate_up_exps,
1907:         /* gate_up_exps_b */ nullptr,
1908:         up_exps_s,
1909:         gate_exps_s,
1910:         down_exps_s,
1911:         selected_experts_in
1912:     );
1913: }
```

## 1.5 `llm_graph_context::build_moe_ffn` — overload B (full implementation, lines 1915–2264)

```cpp
1915: ggml_tensor * llm_graph_context::build_moe_ffn(
1916:          ggml_tensor * cur,
1917:          ggml_tensor * gate_inp,
1918:          ggml_tensor * gate_inp_b,
1919:          ggml_tensor * up_exps,
1920:          ggml_tensor * up_exps_b,
1921:          ggml_tensor * gate_exps,
1922:          ggml_tensor * gate_exps_b,
1923:          ggml_tensor * down_exps,
1924:          ggml_tensor * down_exps_b,
1925:          ggml_tensor * exp_probs_b,
1926:              int64_t   n_expert,
1927:              int64_t   n_expert_used,
1928:      llm_ffn_op_type   type_op,
1929:                 bool   norm_w,
1930:                float   w_scale,
1931:         llama_expert_gating_func_type gating_op,
1932:                  int   il,
1933:          ggml_tensor * probs_in,
1934:          ggml_tensor * gate_up_exps,
1935:          ggml_tensor * gate_up_exps_b,
1936:          ggml_tensor * up_exps_s,
1937:          ggml_tensor * gate_exps_s,
1938:          ggml_tensor * down_exps_s,
1939:          ggml_tensor * selected_experts_in) const {
1940:     const int64_t n_embd   = cur->ne[0];
1941:     const int64_t n_tokens = cur->ne[1];
1942:     const bool weight_before_ffn = arch == LLM_ARCH_LLAMA4; // for llama4, we apply the sigmoid-ed weights before the FFN
1943: 
1944:     ggml_tensor * logits = nullptr;
1945: 
1946:     if (probs_in == nullptr) {
1947:         logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
1948:         if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
1949:             ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
1950:         }
1951:         cb(logits, "ffn_moe_logits", il);
1952:     } else {
1953:         logits = probs_in;
1954:     }
1955: 
1956:     if (gate_inp_b) {
1957:         logits = ggml_add(ctx0, logits, gate_inp_b);
1958:         cb(logits, "ffn_moe_logits_biased", il);
1959:     }
1960: 
1961:     ggml_tensor * probs = nullptr;
1962:     switch (gating_op) {
1963:         case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX:
1964:             {
1965:                 probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
1966:             } break;
1967:         case LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID:
1968:             {
1969:                 probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
1970:             } break;
1971:         case LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT:
1972:             {
1973:                 probs = logits; // [n_expert, n_tokens]
1974:             } break;
1975:         case LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS:
1976:             {
1977:                 probs = ggml_sqrt(ctx0, ggml_softplus(ctx0, logits)); // [n_expert, n_tokens]
1978:             } break;
1979:         default:
1980:             GGML_ABORT("fatal error");
1981:     }
1982:     cb(probs, "ffn_moe_probs", il);
1983: 
1984:     // add experts selection bias - introduced in DeepSeek V3
1985:     // leave probs unbiased as it's later used to get expert weights
1986:     ggml_tensor * selection_probs = probs;
1987:     if (exp_probs_b != nullptr) {
1988:         selection_probs = ggml_add(ctx0, probs, exp_probs_b);
1989:         cb(selection_probs, "ffn_moe_probs_biased", il);
1990:     }
1991: 
1992:     // llama4 doesn't have exp_probs_b, and sigmoid is only used after top_k
1993:     // see: https://github.com/meta-llama/llama-models/blob/699a02993512fb36936b1b0741e13c06790bcf98/models/llama4/moe.py#L183-L198
1994:     if (arch == LLM_ARCH_LLAMA4) {
1995:         selection_probs = logits;
1996:     }
1997: 
1998:     if (arch == LLM_ARCH_GROVEMOE) {
1999:         selection_probs = ggml_sigmoid(ctx0, logits); // [n_expert, n_tokens]
2000:         cb(selection_probs, "ffn_moe_probs_biased", il);
2001:     }
2002: 
2003:     // select top n_group_used expert groups
2004:     // https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/modeling_deepseek.py#L440-L457
2005:     if (hparams.n_expert_groups > 1 && n_tokens > 0) {
2006:         const int64_t n_exp_per_group = n_expert / hparams.n_expert_groups;
2007: 
2008:         // organize experts into n_expert_groups
2009:         ggml_tensor * selection_groups = ggml_reshape_3d(ctx0, selection_probs, n_exp_per_group, hparams.n_expert_groups, n_tokens); // [n_exp_per_group, n_expert_groups, n_tokens]
2010: 
2011:         ggml_tensor * group_scores = ggml_argsort_top_k(ctx0, selection_groups, 2); // [2, n_expert_groups, n_tokens]
2012:         group_scores = ggml_get_rows(ctx0, ggml_reshape_4d(ctx0, selection_groups, 1, selection_groups->ne[0], selection_groups->ne[1], selection_groups->ne[2]), group_scores); // [1, 2, n_expert_groups, n_tokens]
2013: 
2014:         // get top n_group_used expert groups
2015:         group_scores = ggml_sum_rows(ctx0, ggml_reshape_3d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2], group_scores->ne[3])); // [1, n_expert_groups, n_tokens]
2016:         group_scores = ggml_reshape_2d(ctx0, group_scores, group_scores->ne[1], group_scores->ne[2]); // [n_expert_groups, n_tokens]
2017: 
2018:         ggml_tensor * expert_groups = ggml_argsort_top_k(ctx0, group_scores, hparams.n_group_used); // [n_group_used, n_tokens]
2019:         cb(expert_groups, "ffn_moe_group_topk", il);
2020: 
2021:         // mask out the other groups
2022:         selection_probs = ggml_get_rows(ctx0, selection_groups, expert_groups); // [n_exp_per_group, n_group_used, n_tokens]
2023:         selection_probs = ggml_set_rows(ctx0, ggml_fill(ctx0, selection_groups, -INFINITY), selection_probs, expert_groups); // [n_exp_per_group, n_expert_groups, n_tokens]
2024:         selection_probs = ggml_reshape_2d(ctx0, selection_probs, n_expert, n_tokens); // [n_expert, n_tokens]
2025:         cb(selection_probs, "ffn_moe_probs_masked", il);
2026:     }
2027: 
2028:     // select experts
2029:     ggml_tensor * selected_experts = selected_experts_in;
2030:     if (selected_experts == nullptr) {
2031:         selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
2032:         cb(selected_experts->src[0], "ffn_moe_argsort", il);
2033:     }
2034:     cb(selected_experts, "ffn_moe_topk", il);
2035: 
2036:     if (arch == LLM_ARCH_GROVEMOE && n_expert != hparams.n_expert) {
2037:         // TODO: Use scalar div instead when/if implemented
2038:         ggml_tensor * f_sel = ggml_cast(ctx0, selected_experts, GGML_TYPE_F32);
2039:         selected_experts = ggml_cast(ctx0, ggml_scale(ctx0, f_sel, 1.0f / float(hparams.n_group_experts)), GGML_TYPE_I32);
2040:         probs = ggml_reshape_3d(ctx0, probs, 1, hparams.n_expert, n_tokens);
2041:     } else {
2042:         probs = ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens);
2043:     }
2044: 
2045:     ggml_tensor * weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_used, n_tokens]
2046:     cb(weights, "ffn_moe_weights", il);
2047: 
2048: 
2049:     if (gating_op == LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT) {
2050:         weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
2051:         weights = ggml_soft_max(ctx0, weights); // [n_expert_used, n_tokens]
2052:         weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
2053:         cb(weights, "ffn_moe_weights_softmax", il);
2054:     }
2055: 
2056:     if (norm_w) {
2057:         weights = ggml_reshape_2d(ctx0, weights, n_expert_used, n_tokens);
2058: 
2059:         ggml_tensor * weights_sum = ggml_sum_rows(ctx0, weights); // [1, n_tokens]
2060:         cb(weights_sum, "ffn_moe_weights_sum", il);
2061: 
2062:         // Avoid division by zero, clamp to smallest number representable by F16
2063:         weights_sum = ggml_clamp(ctx0, weights_sum, 6.103515625e-5, INFINITY);
2064:         cb(weights_sum, "ffn_moe_weights_sum_clamped", il);
2065: 
2066:         weights = ggml_div(ctx0, weights, weights_sum); // [n_expert_used, n_tokens]
2067:         cb(weights, "ffn_moe_weights_norm", il);
2068: 
2069:         weights = ggml_reshape_3d(ctx0, weights, 1, n_expert_used, n_tokens);
2070:     }
2071:     if (w_scale != 0.0f && w_scale != 1.0f) {
2072:         weights = ggml_scale(ctx0, weights, w_scale);
2073:         cb(weights, "ffn_moe_weights_scaled", il);
2074:     }
2075: 
2076:     //call early so that topk-moe can be used
2077:     ggml_build_forward_expand(gf, weights);
2078: 
2079:     cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
2080: 
2081:     if (weight_before_ffn) {
2082:         // repeat cur to [n_embd, n_expert_used, n_tokens]
2083:         ggml_tensor * repeated = ggml_repeat_4d(ctx0, cur, n_embd, n_expert_used, n_tokens, 1);
2084:         cur = ggml_mul(ctx0, repeated, weights);
2085:         cb(cur, "ffn_moe_weighted", il);
2086:     }
2087: 
2088:     ggml_tensor * up = nullptr;
2089:     ggml_tensor * experts = nullptr;
2090: 
2091:     if (gate_up_exps) {
2092:         // merged gate_up path: one mul_mat_id, then split into gate and up views
2093:         ggml_tensor * gate_up = build_lora_mm_id(gate_up_exps, cur, selected_experts, up_exps_s); // [n_ff*2, n_expert_used, n_tokens]
2094:         cb(gate_up, "ffn_moe_gate_up", il);
2095: 
2096:         if (up_exps_s) {
2097:             cb(gate_up, "ffn_moe_gate_up_scaled", il);
2098:         }
2099: 
2100:         if (gate_up_exps_b) {
2101:             gate_up = ggml_add_id(ctx0, gate_up, gate_up_exps_b, selected_experts);
2102:             cb(gate_up, "ffn_moe_gate_up_biased", il);
2103:         }
2104: 
2105:         const int64_t n_ff = gate_up->ne[0] / 2;
2106:         cur = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
2107:         cb(cur, "ffn_moe_gate", il);
2108:         up  = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
2109:         cb(up, "ffn_moe_up", il);
2110:     } else {
2111:         // separate gate and up path
2112:         up = build_lora_mm_id(up_exps, cur, selected_experts, up_exps_s); // [n_ff, n_expert_used, n_tokens]
2113:         cb(up, "ffn_moe_up", il);
2114: 
2115:         if (up_exps_s) {
2116:             cb(up, "ffn_moe_up_scaled", il);
2117:         }
2118: 
2119:         if (up_exps_b) {
2120:             up = ggml_add_id(ctx0, up, up_exps_b, selected_experts);
2121:             cb(up, "ffn_moe_up_biased", il);
2122:         }
2123: 
2124:         if (gate_exps) {
2125:             cur = build_lora_mm_id(gate_exps, cur, selected_experts, gate_exps_s); // [n_ff, n_expert_used, n_tokens]
2126:             cb(cur, "ffn_moe_gate", il);
2127:         } else {
2128:             cur = up;
2129:         }
2130: 
2131:         if (gate_exps_s) {
2132:             cb(cur, "ffn_moe_gate_scaled", il);
2133:         }
2134: 
2135:         if (gate_exps_b) {
2136:             cur = ggml_add_id(ctx0, cur, gate_exps_b, selected_experts);
2137:             cb(cur, "ffn_moe_gate_biased", il);
2138:         }
2139:     }
2140: 
2141:     const bool has_gate = gate_exps || gate_up_exps;
2142: 
2143:     switch (type_op) {
2144:         case LLM_FFN_SILU:
2145:             if (gate_exps) {
2146:                 if (il >= 0) {
2147:                     const float limit = hparams.swiglu_clamp_exp[il];
2148:                     constexpr float eps = 1e-6f;
2149:                     if (limit > eps) {
2150:                         up = ggml_clamp(ctx0, up, -limit, limit);
2151:                         cb(up, "ffn_moe_up_clamped", il);
2152: 
2153:                         if (arch == LLM_ARCH_DEEPSEEK4 || (arch == LLM_ARCH_DFLASH && hparams.dsv4_hc_mult > 0)) {
2154:                             cur = ggml_clamp(ctx0, cur, -INFINITY, limit);
2155:                             cb(cur, "ffn_moe_gate_clamped", il);
2156:                             cur = ggml_swiglu_split(ctx0, cur, up);
2157:                         } else {
2158:                             ggml_tensor * gate_act = ggml_silu(ctx0, cur);
2159:                             cb(gate_act, "ffn_moe_silu", il);
2160:                             gate_act = ggml_clamp(ctx0, gate_act, -INFINITY, limit);
2161:                             cb(gate_act, "ffn_moe_silu_clamped", il);
2162:                             cur = ggml_mul(ctx0, gate_act, up);
2163:                         }
2164:                         cb(cur, "ffn_moe_swiglu_limited", il);
2165:                         break;
2166:                     }
2167:                 }
2168:             }
2169: 
2170:             if (has_gate) {
2171:                 cur = ggml_swiglu_split(ctx0, cur, up);
2172:                 cb(cur, "ffn_moe_swiglu", il);
2173:             } else {
2174:                 cur = ggml_silu(ctx0, cur);
2175:                 cb(cur, "ffn_moe_silu", il);
2176:             } break;
2177:         case LLM_FFN_GELU:
2178:             if (has_gate) {
2179:                 cur = ggml_geglu_split(ctx0, cur, up);
2180:                 cb(cur, "ffn_moe_geglu", il);
2181:             } else {
2182:                 cur = ggml_gelu(ctx0, cur);
2183:                 cb(cur, "ffn_moe_gelu", il);
2184:             } break;
2185:         case LLM_FFN_SWIGLU_OAI_MOE:
2186:             {
2187:                 // TODO: move to hparams?
2188:                 constexpr float alpha = 1.702f;
2189:                 constexpr float limit = 7.0f;
2190:                 cur = ggml_swiglu_oai(ctx0, cur, up, alpha, limit);
2191:                 cb(cur, "ffn_moe_swiglu_oai", il);
2192:             } break;
2193:         case LLM_FFN_RELU:
2194:             if (has_gate) {
2195:                 cur = ggml_reglu_split(ctx0, cur, up);
2196:                 cb(cur, "ffn_moe_reglu", il);
2197:             } else {
2198:                 cur = ggml_relu(ctx0, cur);
2199:                 cb(cur, "ffn_moe_relu", il);
2200:             } break;
2201:         case LLM_FFN_RELU_SQR:
2202:             if (has_gate) {
2203:                 // TODO: add support for gated squared relu
2204:                 GGML_ABORT("fatal error: gated squared relu not implemented");
2205:             } else {
2206:                 cur = ggml_relu(ctx0, cur);
2207:                 cur = ggml_sqr(ctx0, cur);
2208:                 cb(cur, "ffn_moe_relu_sqr", il);
2209:             } break;
2210:         default:
2211:             GGML_ABORT("fatal error");
2212:     }
2213: 
2214:     experts = build_lora_mm_id(down_exps, cur, selected_experts, down_exps_s); // [n_embd, n_expert_used, n_tokens]
2215:     cb(experts, "ffn_moe_down", il);
2216: 
2217:     if (down_exps_s) {
2218:         cb(experts, "ffn_moe_down_scaled", il);
2219:     }
2220: 
2221:     if (down_exps_b) {
2222:         experts = ggml_add_id(ctx0, experts, down_exps_b, selected_experts);
2223:         cb(experts, "ffn_moe_down_biased", il);
2224:     }
2225: 
2226:     if (!weight_before_ffn) {
2227:         experts = ggml_mul(ctx0, experts, weights);
2228:         cb(experts, "ffn_moe_weighted", il);
2229:     }
2230: 
2231:     ggml_build_forward_expand(gf, experts);
2232: 
2233:     ggml_tensor * cur_experts[LLAMA_MAX_EXPERTS] = { nullptr };
2234: 
2235:     assert(n_expert_used > 0);
2236: 
2237:     // order the views before the adds
2238:     for (uint32_t i = 0; i < hparams.n_expert_used; ++i) {
2239:         cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1]);
2240: 
2241:         ggml_build_forward_expand(gf, cur_experts[i]);
2242:     }
2243: 
2244:     // aggregate experts
2245:     // note: here we explicitly use hparams.n_expert_used instead of n_expert_used
2246:     //       to avoid potentially a large number of add nodes during warmup
2247:     //       ref: https://github.com/ggml-org/llama.cpp/pull/14753
2248:     ggml_tensor * moe_out = cur_experts[0];
2249: 
2250:     for (uint32_t i = 1; i < hparams.n_expert_used; ++i) {
2251:         moe_out = ggml_add(ctx0, moe_out, cur_experts[i]);
2252: 
2253:         ggml_build_forward_expand(gf, moe_out);
2254:     }
2255: 
2256:     if (hparams.n_expert_used == 1) {
2257:         // avoid returning a non-contiguous tensor
2258:         moe_out = ggml_cont(ctx0, moe_out);
2259:     }
2260: 
2261:     cb(moe_out, "ffn_moe_out", il);
2262: 
2263:     return moe_out;
2264: }
```

**Key semantics of `build_moe_ffn`:**
- Gating: `logits = gate_inp @ cur` → `probs = softmax(logits)` for `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX` (the Qwen3.5 case).
- `selection_probs = probs + exp_probs_b` (bias kept separate; probs stay unbiased for weight extraction).
- Top-K: `selected_experts = argsort_top_k(selection_probs, n_expert_used)` → shape `[n_expert_used, n_tokens]`.
- Weights: `weights = get_rows(probs[1, n_expert, n_tokens], selected_experts)` → `[1, n_expert_used, n_tokens]`.
- `norm_w`: `weights = weights / clamp(sum_rows(weights), 6.103515625e-5, INFINITY)` (F16 min normal), then reshaped to `[1, n_expert_used, n_tokens]`.
- `w_scale` multiplies weights if not 0.0/1.0.
- Input reshaped to `[n_embd, 1, n_tokens]`; each expert matmul uses `ggml_mul_mat_id` (batched per-expert) via `build_lora_mm_id`.
- Gate+up: either merged `gate_up_exps` (single `[n_ff*2, ...]` then split views) or separate `up_exps`/`gate_exps`.
- Activation: `LLM_FFN_SILU` → `cur = swiglu_split(gate, up)` when gated.
- Down: `experts = down_exps @ cur`, then `experts *= weights` (unless `weight_before_ffn`), then views `[n_embd, n_tokens]` per expert are summed into `moe_out`.

## 1.6 `build_attn_inp_kq_mask` (lines 28–45)

```cpp
28: static ggml_tensor * build_attn_inp_kq_mask(
29:         ggml_context * ctx,
30:         const llama_kv_cache_context * mctx,
31:         const llama_ubatch & ubatch,
32:         const llama_cparams & cparams) {
33:     const auto n_kv     = mctx->get_n_kv();
34:     const auto n_tokens = ubatch.n_tokens;
35:     const auto n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq;
36: 
37:     // flash attention requires an f16 mask
38:     const auto type = cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32;
39: 
40:     ggml_tensor * res = ggml_new_tensor_4d(ctx, type, n_kv, n_tokens/n_stream, 1, n_stream);
41:     ggml_set_input(res);
42:     ggml_set_name(res, "attn_inp_kq_mask");
43: 
44:     return res;
45: }
```

## 1.7 `build_attn_mha` (lines 2500–2633)

```cpp
2500: ggml_tensor * llm_graph_context::build_attn_mha(
2501:          ggml_tensor * q,
2502:          ggml_tensor * k,
2503:          ggml_tensor * v,
2504:          ggml_tensor * kq_b,
2505:          ggml_tensor * kq_mask,
2506:          ggml_tensor * sinks,
2507:          ggml_tensor * v_mla,
2508:                float   kq_scale,
2509:                  int   il) const {
2510:     const bool v_trans = v->nb[1] > v->nb[2];
2511: 
2512:     // split the batch into streams if needed
2513:     const auto n_stream = k->ne[3];
2514: 
2515:     q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream, q->nb[1], q->nb[2], q->nb[3]/n_stream, 0);
2516: 
2517:     q = ggml_permute(ctx0, q, 0, 2, 1, 3);
2518:     k = ggml_permute(ctx0, k, 0, 2, 1, 3);
2519:     v = ggml_permute(ctx0, v, 0, 2, 1, 3);
2520: 
2521:     ggml_tensor * cur;
2522: 
2523:     const bool use_flash_attn = cparams.flash_attn && kq_b == nullptr;
2524:     if (use_flash_attn) {
2525:         GGML_ASSERT(kq_b == nullptr && "Flash attention does not support KQ bias yet");
2526: 
2527:         if (v_trans) {
2528:             v = ggml_transpose(ctx0, v);
2529:         }
2530: 
2531:         // this can happen when KV cache is not used (e.g. an embedding model with non-causal attn)
2532:         if (k->type == GGML_TYPE_F32) {
2533:             k = ggml_cast(ctx0, k, GGML_TYPE_F16);
2534:         }
2535: 
2536:         if (v->type == GGML_TYPE_F32) {
2537:             v = ggml_cast(ctx0, v, GGML_TYPE_F16);
2538:         }
2539: 
2540:         cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
2541:                                   hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
2542:         res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});
2543: 
2544:         ggml_flash_attn_ext_add_sinks(cur, sinks);
2545:         ggml_flash_attn_ext_set_prec (cur, GGML_PREC_F32);
2546: 
2547:         if (v_mla) {
2548: #if 0
2549:             // v_mla can be applied as a matrix-vector multiplication with broadcasting across dimension 3 == n_tokens.
2550:             // However, the code is optimized for dimensions 0 and 1 being large, so this is inefficient.
2551:             cur = ggml_reshape_4d(ctx0, cur, v_mla->ne[0], 1, n_head, n_tokens);
2552:             cur = ggml_mul_mat(ctx0, v_mla, cur);
2553: #else
2554:             // It's preferable to do the calculation as a matrix-matrix multiplication with n_tokens in dimension 1.
2555:             // The permutations are noops and only change how the tensor data is interpreted.
2556:             cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
2557:             cur = ggml_mul_mat(ctx0, v_mla, cur);
2558:             cb(cur, "fattn_mla", il);
2559:             cur = ggml_permute(ctx0, cur, 0, 2, 1, 3);
2560:             cur = ggml_cont(ctx0, cur); // Needed because ggml_reshape_2d expects contiguous inputs.
2561: #endif
2562:         }
2563: 
2564:         cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
2565:     } else {
2566:         ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
2567:         cb(kq, "kq", il);
2568: 
2569:         // note: this op tends to require high floating point range
2570:         //       while for some models F16 is enough, for others it is not, so we default to F32 here
2571:         ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
2572: 
2573:         if (arch == LLM_ARCH_GROK) {
2574:             // need to do the following:
2575:             // multiply by attn_output_multiplier
2576:             // and then :
2577:             // kq = 30 * tanh(kq / 30)
2578:             // before the softmax below
2579: 
2580:             kq = ggml_tanh(ctx0, ggml_scale(ctx0, kq, hparams.f_attn_out_scale / hparams.f_attn_logit_softcapping));
2581:             cb(kq, "kq_tanh", il);
2582:             kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
2583:             cb(kq, "kq_scaled", il);
2584:         }
2585: 
2586:         if (hparams.attn_soft_cap) {
2587:             kq = ggml_scale(ctx0, kq, 1.0f / hparams.f_attn_logit_softcapping);
2588:             cb(kq, "kq_scaled_1", il);
2589:             kq = ggml_tanh (ctx0, kq);
2590:             cb(kq, "kq_tanh", il);
2591:             kq = ggml_scale(ctx0, kq, hparams.f_attn_logit_softcapping);
2592:             cb(kq, "kq_scaled_2", il);
2593:         }
2594: 
2595:         if (kq_b) {
2596:             kq = ggml_add(ctx0, kq, kq_b);
2597:             cb(kq, "kq_plus_kq_b", il);
2598:         }
2599: 
2600:         kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
2601:         ggml_soft_max_add_sinks(kq, sinks);
2602:         cb(kq, "kq_soft_max", il);
2603: 
2604:         if (!v_trans) {
2605:             // note: avoid this branch
2606:             v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
2607:             cb(v, "v_cont", il);
2608:         }
2609: 
2610:         ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
2611:         cb(kqv, "kqv", il);
2612: 
2613:         // for MLA with the absorption optimization, we need to "decompress" from MQA back to MHA
2614:         if (v_mla) {
2615:             kqv = ggml_mul_mat(ctx0, v_mla, kqv);
2616:             cb(kqv, "kqv_mla", il);
2617:         }
2618: 
2619:         cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
2620: 
2621:         // recombine streams
2622:         cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
2623: 
2624:         if (!cparams.offload_kqv) {
2625:             // all nodes between the KV store and the attention output are run on the CPU
2626:             ggml_backend_sched_set_tensor_backend(sched, cur, backend_cpu);
2627:         }
2628:     }
2629: 
2630:     ggml_build_forward_expand(gf, cur);
2631: 
2632:     return cur;
2633: }
```

**Semantics:** `kq = k^T q`, optional `kq_b` add, then `softmax_ext(kq, kq_mask, kq_scale, max_alibi_bias)` (kq_scale applied inside softmax), `kqv = v kq`, optional `v_mla` decompress, permute back and `cont_2d`.

## 1.8 `build_attn` — all 7 overloads

### 1.8a `llm_graph_input_attn_no_cache` (lines 2660–2710)
```cpp
2660: ggml_tensor * llm_graph_context::build_attn(
2661:         llm_graph_input_attn_no_cache * inp,
2662:         ggml_tensor * wo,
2663:         ggml_tensor * wo_b,
2664:         ggml_tensor * wo_s,
2665:         ggml_tensor * q_cur,
2666:         ggml_tensor * k_cur,
2667:         ggml_tensor * v_cur,
2668:         ggml_tensor * kq_b,
2669:         ggml_tensor * sinks,
2670:         ggml_tensor * v_mla,
2671:             float     kq_scale,
2672:             int       il) const {
2673:     GGML_UNUSED(n_tokens);
2674: 
2675:     // these nodes are added to the graph together so that they are not reordered
2676:     // by doing so, the number of splits in the graph is reduced
2677:     ggml_build_forward_expand(gf, q_cur);
2678:     ggml_build_forward_expand(gf, k_cur);
2679:     ggml_build_forward_expand(gf, v_cur);
2680: 
2681:     const bool is_swa = hparams.is_swa(il);
2682: 
2683:     const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();
2684: 
2685:     // [TAG_NO_CACHE_PAD]
2686:     // TODO: if ubatch.equal_seqs() == true, we can split the three tensors below into ubatch.n_seqs_unq streams
2687:     //       but it might not be worth it: https://github.com/ggml-org/llama.cpp/pull/15636
2688:     //assert(!ubatch.equal_seqs() || (k_cur->ne[3] == 1 && k_cur->ne[3] == ubatch.n_seqs_unq));
2689: 
2690:     ggml_tensor * q = q_cur;
2691:     ggml_tensor * k = k_cur;
2692:     ggml_tensor * v = v_cur;
2693: 
2694:     ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
2695:     cb(cur, "kqv_out", il);
2696: 
2697:     if (wo) {
2698:         cur = build_lora_mm(wo, cur, wo_s);
2699:     }
2700: 
2701:     if (wo_b) {
2702:         //cb(cur, "kqv_wo", il);
2703:     }
2704: 
2705:     if (wo_b) {
2706:         cur = ggml_add(ctx0, cur, wo_b);
2707:     }
2708: 
2709:     return cur;
2710: }
```

### 1.8b `llm_graph_input_attn_kv` (lines 2745–2818) — the one used by `llm_build_qwen3_5::build_layer_attn`
```cpp
2745: ggml_tensor * llm_graph_context::build_attn(
2746:         llm_graph_input_attn_kv * inp,
2747:         ggml_tensor * wo,
2748:         ggml_tensor * wo_b,
2749:         ggml_tensor * wo_s,
2750:         ggml_tensor * q_cur,
2751:         ggml_tensor * k_cur,
2752:         ggml_tensor * v_cur,
2753:         ggml_tensor * kq_b,
2754:         ggml_tensor * sinks,
2755:         ggml_tensor * v_mla, // TODO: remove
2756:             float     kq_scale,
2757:             int       il) const {
2758:     GGML_ASSERT(v_mla == nullptr);
2759: 
2760:     if (inp->self_k_rot) {
2761:         q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
2762:         k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
2763:     }
2764: 
2765:     if (inp->self_v_rot) {
2766:         v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
2767:     }
2768: 
2769:     // these nodes are added to the graph together so that they are not reordered
2770:     // by doing so, the number of splits in the graph is reduced
2771:     // expand k later to enable rope fusion which directly writes into k-v cache
2772:     ggml_build_forward_expand(gf, q_cur);
2773:     ggml_build_forward_expand(gf, v_cur);
2774:     ggml_build_forward_expand(gf, k_cur);
2775: 
2776:     const auto * mctx_cur = inp->mctx;
2777: 
2778:     // store to KV cache
2779:     {
2780:         const auto & k_idxs = inp->get_k_idxs();
2781:         const auto & v_idxs = inp->get_v_idxs();
2782: 
2783:         ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
2784:         ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
2785:     }
2786: 
2787:     ggml_tensor * kq_mask = inp->get_kq_mask();
2788: 
2789:     ggml_tensor * q = q_cur;
2790:     ggml_tensor * k = mctx_cur->get_k(ctx0, il);
2791:     ggml_tensor * v = mctx_cur->get_v(ctx0, il);
2792: 
2793:     ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
2794:     cb(cur, "kqv_out", il);
2795: 
2796:     if (inp->self_v_rot) {
2797:         cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
2798:     }
2799: 
2800:     if (wo) {
2801:         if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE || arch == LLM_ARCH_JAIS2) {
2802:             // GLM4, GLM4_MOE, and JAIS2 seem to have numerical issues with half-precision accumulators
2803:             cur = build_lora_mm(wo, cur);
2804:             ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
2805:             if (wo_s) {
2806:                 cur = ggml_mul(ctx0, cur, wo_s);
2807:             }
2808:         } else {
2809:             cur = build_lora_mm(wo, cur, wo_s);
2810:         }
2811:     }
2812: 
2813:     if (wo_b) {
2814:         cur = ggml_add(ctx0, cur, wo_b);
2815:     }
2816: 
2817:     return cur;
2818: }
```

### 1.8c `llm_graph_input_attn_k` (lines 2849–2905)
```cpp
2849: ggml_tensor * llm_graph_context::build_attn(
2850:         llm_graph_input_attn_k * inp,
2851:         ggml_tensor * wo,
2852:         ggml_tensor * wo_b,
2853:         ggml_tensor * wo_s,
2854:         ggml_tensor * q_cur,
2855:         ggml_tensor * k_cur,
2856:         ggml_tensor * v_cur,
2857:         ggml_tensor * kq_b,
2858:         ggml_tensor * sinks,
2859:         ggml_tensor * v_mla,
2860:             float     kq_scale,
2861:             int       il) const {
2862:     // these nodes are added to the graph together so that they are not reordered
2863:     // by doing so, the number of splits in the graph is reduced
2864:     // expand k later to enable rope fusion which directly writes into k-v cache
2865:     ggml_build_forward_expand(gf, q_cur);
2866:     ggml_build_forward_expand(gf, v_cur);
2867:     ggml_build_forward_expand(gf, k_cur);
2868: 
2869:     const auto * mctx_cur = inp->mctx;
2870: 
2871:     // store to KV cache
2872:     {
2873:         const auto & k_idxs = inp->get_k_idxs();
2874: 
2875:         ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
2876:     }
2877: 
2878:     const auto & kq_mask = inp->get_kq_mask();
2879: 
2880:     ggml_tensor * q = q_cur;
2881:     ggml_tensor * k = mctx_cur->get_k(ctx0, il);
2882:     ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);
2883: 
2884:     ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
2885:     cb(cur, "kqv_out", il);
2886: 
2887:     if (wo) {
2888:         if (arch == LLM_ARCH_GLM4 || arch == LLM_ARCH_GLM4_MOE) {
2889:             // GLM4 and GLM4_MOE seem to have numerical issues with half-precision accumulators
2890:             cur = build_lora_mm(wo, cur);
2891:             ggml_mul_mat_set_prec(cur, GGML_PREC_F32);
2892:             if (wo_s) {
2893:                 cur = ggml_mul(ctx0, cur, wo_s);
2894:             }
2895:         } else {
2896:             cur = build_lora_mm(wo, cur, wo_s);
2897:         }
2898:     }
2899: 
2900:     if (wo_b) {
2901:         cur = ggml_add(ctx0, cur, wo_b);
2902:     }
2903: 
2904:     return cur;
2905: }
```

### 1.8d `llm_graph_input_attn_k_dsa` (lines 2907–2981)
```cpp
2907: ggml_tensor * llm_graph_context::build_attn(
2908:         llm_graph_input_attn_k_dsa * inp,
2909:         ggml_tensor * wo,
2910:         ggml_tensor * wo_b,
2911:         ggml_tensor * wo_s,
2912:         ggml_tensor * q_cur,
2913:         ggml_tensor * k_cur,
2914:         ggml_tensor * v_cur,
2915:         ggml_tensor * kq_b,
2916:         ggml_tensor * sinks,
2917:         ggml_tensor * v_mla,
2918:         ggml_tensor * top_k,
2919:             float     kq_scale,
2920:             int       il) const {
2921:     // these nodes are added to the graph together so that they are not reordered
2922:     // by doing so, the number of splits in the graph is reduced
2923:     // expand k later to enable rope fusion which directly writes into k-v cache
2924:     ggml_build_forward_expand(gf, q_cur);
2925:     ggml_build_forward_expand(gf, v_cur);
2926:     ggml_build_forward_expand(gf, k_cur);
2927: 
2928:     const auto * mctx_cur = inp->mctx->get_mla();
2929: 
2930:     // store to KV cache
2931:     {
2932:         const auto & k_idxs = inp->get_k_idxs_mla();
2933: 
2934:         ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
2935:     }
2936: 
2937:     const auto & kq_mask = inp->get_kq_mask_mla();
2938: 
2939:     // prepare new kq mask - starts filled with -INFINITY
2940:     ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);
2941: 
2942:     // reshape KQ mask into tensor with rows of size 1:
2943:     // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
2944:     kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);
2945: 
2946:     // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
2947:     ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);
2948: 
2949:     // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
2950:     // this will be our source of zero values for unmasking top k mask elements
2951:     ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
2952:     zeros = ggml_fill(ctx0, zeros, 0.0f);
2953: 
2954:     // modify KQ mask by unmasking elements that are in top_k indices
2955:     // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
2956:     ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);
2957: 
2958:     // reshape to restore the original shape of KQ mask:
2959:     // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
2960:     kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);
2961: 
2962:     // combine with the original kq mask
2963:     kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);
2964: 
2965:     ggml_tensor * q = q_cur;
2966:     ggml_tensor * k = mctx_cur->get_k(ctx0, il);
2967:     ggml_tensor * v = ggml_view_4d(ctx0, k, v_cur->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[1], k->nb[2], k->nb[3], 0);
2968: 
2969:     ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask_top_k, sinks, v_mla, kq_scale, il);
2970:     cb(cur, "kqv_out", il);
2971: 
2972:     if (wo) {
2973:         cur = build_lora_mm(wo, cur, wo_s);
2974:     }
2975: 
2976:     if (wo_b) {
2977:         cur = ggml_add(ctx0, cur, wo_b);
2978:     }
2979: 
2980:     return cur;
2981: }
```

### 1.8e `llm_graph_input_attn_kv_iswa` (lines 2983–3068)
```cpp
2983: ggml_tensor * llm_graph_context::build_attn(
2984:         llm_graph_input_attn_kv_iswa * inp,
2985:         ggml_tensor * wo,
2986:         ggml_tensor * wo_b,
2987:         ggml_tensor * wo_s,
2988:         ggml_tensor * q_cur,
2989:         ggml_tensor * k_cur,
2990:         ggml_tensor * v_cur,
2991:         ggml_tensor * kq_b,
2992:         ggml_tensor * sinks,
2993:         ggml_tensor * v_mla,
2994:             float     kq_scale,
2995:             int       il) const {
2996:     const bool is_swa = hparams.is_swa(il);
2997: 
2998:     auto * k_rot = is_swa ? inp->self_k_rot_swa : inp->self_k_rot;
2999:     auto * v_rot = is_swa ? inp->self_v_rot_swa : inp->self_v_rot;
3000: 
3001:     if (k_rot) {
3002:         q_cur = llama_mul_mat_hadamard(ctx0, q_cur, k_rot);
3003:         if (k_cur) {
3004:             k_cur = llama_mul_mat_hadamard(ctx0, k_cur, k_rot);
3005:         }
3006:     }
3007:     if (v_rot) {
3008:         if (v_cur) {
3009:             v_cur = llama_mul_mat_hadamard(ctx0, v_cur, v_rot);
3010:         }
3011:     }
3012: 
3013:     // these nodes are added to the graph together so that they are not reordered
3014:     // by doing so, the number of splits in the graph is reduced
3015:     ggml_build_forward_expand(gf, q_cur);
3016: 
3017:     if (k_cur) {
3018:         ggml_build_forward_expand(gf, k_cur);
3019:     }
3020: 
3021:     if (v_cur) {
3022:         ggml_build_forward_expand(gf, v_cur);
3023:     }
3024: 
3025:     const auto * mctx_iswa = inp->mctx;
3026: 
3027:     const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();
3028: 
3029:     // optionally store to KV cache
3030:     if (k_cur) {
3031:         const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();
3032: 
3033:         ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
3034:     }
3035: 
3036:     if (v_cur) {
3037:         const auto & v_idxs = is_swa ? inp->get_v_idxs_swa() : inp->get_v_idxs();
3038: 
3039:         ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
3040:     }
3041: 
3042:     const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();
3043: 
3044:     ggml_tensor * q = q_cur;
3045:     ggml_tensor * k = mctx_cur->get_k(ctx0, il);
3046:     ggml_tensor * v = mctx_cur->get_v(ctx0, il);
3047: 
3048:     ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
3049:     cb(cur, "kqv_out", il);
3050: 
3051:     if (v_rot) {
3052:         cur = llama_mul_mat_hadamard(ctx0, cur, v_rot);
3053:     }
3054: 
3055:     if (wo) {
3056:         cur = build_lora_mm(wo, cur, wo_s);
3057:     }
3058: 
3059:     if (wo_b) {
3060:         //cb(cur, "kqv_wo", il);
3061:     }
3062: 
3063:     if (wo_b) {
3064:         cur = ggml_add(ctx0, cur, wo_b);
3065:     }
3066: 
3067:     return cur;
3068: }
```

### 1.8f `llm_graph_input_attn_k_iswa` (lines 3070–3137)
```cpp
3070: ggml_tensor * llm_graph_context::build_attn(
3071:         llm_graph_input_attn_k_iswa * inp,
3072:         ggml_tensor * wo,
3073:         ggml_tensor * wo_b,
3074:         ggml_tensor * wo_s,
3075:         ggml_tensor * q_cur,
3076:         ggml_tensor * k_cur,
3077:         ggml_tensor * v_cur,
3078:         ggml_tensor * kq_b,
3079:         ggml_tensor * sinks,
3080:         ggml_tensor * v_mla,
3081:             float     kq_scale,
3082:             int       il) const {
3083:     const bool is_swa = hparams.is_swa(il);
3084: 
3085:     GGML_UNUSED(v_cur);
3086: 
3087:     auto * k_rot = is_swa ? inp->self_k_rot_swa : inp->self_k_rot;
3088: 
3089:     if (k_rot) {
3090:         q_cur = llama_mul_mat_hadamard(ctx0, q_cur, k_rot);
3091:         if (k_cur) {
3092:             k_cur = llama_mul_mat_hadamard(ctx0, k_cur, k_rot);
3093:         }
3094:     }
3095: 
3096:     // these nodes are added to the graph together so that they are not reordered
3097:     // by doing so, the number of splits in the graph is reduced
3098:     ggml_build_forward_expand(gf, q_cur);
3099: 
3100:     if (k_cur) {
3101:         ggml_build_forward_expand(gf, k_cur);
3102:     }
3103: 
3104:     const auto * mctx_iswa = inp->mctx;
3105:     const auto * mctx_cur = is_swa ? mctx_iswa->get_swa() : mctx_iswa->get_base();
3106: 
3107:     // optionally store to KV cache
3108:     if (k_cur) {
3109:         const auto & k_idxs = is_swa ? inp->get_k_idxs_swa() : inp->get_k_idxs();
3110: 
3111:         ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
3112:     }
3113: 
3114:     const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();
3115: 
3116:     // MLA-style attention: the cached K is used as V
3117:     ggml_tensor * q = q_cur;
3118:     ggml_tensor * k = mctx_cur->get_k(ctx0, il);
3119:     ggml_tensor * v = k;
3120: 
3121:     ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
3122:     cb(cur, "kqv_out", il);
3123: 
3124:     if (k_rot) {
3125:         cur = llama_mul_mat_hadamard(ctx0, cur, k_rot);
3126:     }
3127: 
3128:     if (wo) {
3129:         cur = build_lora_mm(wo, cur, wo_s);
3130:     }
3131: 
3132:     if (wo_b) {
3133:         cur = ggml_add(ctx0, cur, wo_b);
3134:     }
3135: 
3136:     return cur;
3137: }
```

### 1.8g `llm_graph_input_attn_cross` (lines 3155–3196)
```cpp
3155: ggml_tensor * llm_graph_context::build_attn(
3156:         llm_graph_input_attn_cross * inp,
3157:         ggml_tensor * wo,
3158:         ggml_tensor * wo_b,
3159:         ggml_tensor * wo_s,
3160:         ggml_tensor * q_cur,
3161:         ggml_tensor * k_cur,
3162:         ggml_tensor * v_cur,
3163:         ggml_tensor * kq_b,
3164:         ggml_tensor * sinks,
3165:         ggml_tensor * v_mla,
3166:             float     kq_scale,
3167:             int       il) const {
3168:     // these nodes are added to the graph together so that they are not reordered
3169:     // by doing so, the number of splits in the graph is reduced
3170:     ggml_build_forward_expand(gf, q_cur);
3171:     ggml_build_forward_expand(gf, k_cur);
3172:     ggml_build_forward_expand(gf, v_cur);
3173: 
3174:     const auto & kq_mask = inp->get_kq_mask_cross();
3175: 
3176:     ggml_tensor * q = q_cur;
3177:     ggml_tensor * k = k_cur;
3178:     ggml_tensor * v = v_cur;
3179: 
3180:     ggml_tensor * cur = build_attn_mha(q, k, v, kq_b, kq_mask, sinks, v_mla, kq_scale, il);
3181:     cb(cur, "kqv_out", il);
3182: 
3183:     if (wo) {
3184:         cur = build_lora_mm(wo, cur, wo_s);
3185:     }
3186: 
3187:     if (wo_b) {
3188:         //cb(cur, "kqv_wo", il);
3189:     }
3190: 
3191:     if (wo_b) {
3192:         cur = ggml_add(ctx0, cur, wo_b);
3193:     }
3194: 
3195:     return cur;
3196: }
```

## 1.9 `llm_graph_context` constructor (lines 1428–1471) — shows `n_rot` and `rope_type` member init

```cpp
1428: llm_graph_context::llm_graph_context(const llm_graph_params & params) :
1429:     arch             (params.arch),
1430:     hparams          (params.hparams),
1431:     cparams          (params.cparams),
1432:     ubatch           (params.ubatch),
1433:     n_embd           (hparams.n_embd),
1434:     n_layer          (hparams.n_layer()),
1435:     n_layer_nextn    (hparams.n_layer_nextn),
1436:     n_rot            (hparams.n_rot()),
1437:     n_ctx            (cparams.n_ctx),
1438:     n_head           (hparams.n_head()),
1439:     n_head_kv        (hparams.n_head_kv()),
1440:     n_embd_head_k    (hparams.n_embd_head_k()),
1441:     n_embd_k_gqa     (hparams.n_embd_k_gqa()),
1442:     n_embd_head_v    (hparams.n_embd_head_v()),
1443:     n_embd_v_gqa     (hparams.n_embd_v_gqa()),
1444:     n_expert         (hparams.n_expert),
1445:     n_expert_used    (cparams.warmup ? hparams.n_expert : hparams.n_expert_used),
1446:     freq_base        (cparams.rope_freq_base),
1447:     freq_scale       (cparams.rope_freq_scale),
1448:     ext_factor       (cparams.yarn_ext_factor),
1449:     attn_factor      (cparams.yarn_attn_factor),
1450:     beta_fast        (cparams.yarn_beta_fast),
1451:     beta_slow        (cparams.yarn_beta_slow),
1452:     norm_eps         (hparams.f_norm_eps),
1453:     norm_rms_eps     (hparams.f_norm_rms_eps),
1454:     n_tokens         (ubatch.n_tokens),
1455:     n_outputs        (params.n_outputs),
1456:     n_ctx_orig       (cparams.n_ctx_orig_yarn),
1457:     pooling_type     (cparams.pooling_type),
1458:     rope_type        (hparams.rope_type),
1459:     sched            (params.sched),
1460:     backend_cpu      (params.backend_cpu),
1461:     cvec             (params.cvec),
1462:     loras            (params.loras),
1463:     mctx             (params.mctx),
1464:     cross            (params.cross),
1465:     samplers         (params.samplers),
1466:     cb_func          (params.cb),
1467:     res              (params.res),
1468:     ctx0             (res->get_ctx()),
1469:     gf               (res->get_gf()) {
1470:         res->set_params(params);
1471:     }
```

## 1.10 Rope in file 1 — IMPORTANT FINDING

**There are NO `ggml_rope`, `ggml_rope_ext`, or `ggml_rope_multi` calls anywhere in this version of `llama-graph.cpp`.** A full-content search for `rope|sections|ggml_rope` returns only:
- Line 1436 (`n_rot (hparams.n_rot())`), 1446–1447 (`freq_base`/`freq_scale` from cparams), 1456 (`n_ctx_orig`), 1458 (`rope_type (hparams.rope_type)`) — all in the constructor above.
- Line 2771/2864/2923 — comments "expand k later to enable rope fusion which directly writes into k-v cache".
- `rope_sections` and `ggml_rope_multi` do not appear in this file.

In this refactored graph code, rope is applied via `inp->self_k_rot` (a hadamard product tensor, `llama_mul_mat_hadamard`, see build_attn 1.8b/1.8e/1.8f), and the explicit `ggml_rope_ext` calls for Qwen3.5 live in the **qwen3-5 model builder** (see Supplementary section below).

Also extracted for completeness: `build_qkv` (lines 1592–1666), `build_rs` (3344–3378, 3410–3421) — these are in the file but the `build_rs` bodies are included above in my search output; let me note they are at lines 3344-3378 and 3410-3421.

---

# FILE 2: `/home/shinde/.local/share/opencode/tool-output/tool_fed1805be0011zMzmwOdWpeTy3` — llama.cpp `src/llama-model.cpp` (2975 lines)

## 2.1 `llama_model_rope_type` (lines 2558–2727) — the qwen35moe-specific rope_type

```cpp
2558: llama_rope_type llama_model_rope_type(const llama_model * model) {
2559:     switch (model->arch) {
2560:         // these models do not use RoPE
2561:         case LLM_ARCH_CLIP:
2562:         case LLM_ARCH_GPT2:
2563:         case LLM_ARCH_GPTJ:
2564:         case LLM_ARCH_MPT:
2565:         case LLM_ARCH_REFACT:
2566:         case LLM_ARCH_BLOOM:
2567:         case LLM_ARCH_MAMBA:
2568:         case LLM_ARCH_MAMBA2:
2569:         case LLM_ARCH_JAMBA:
2570:         case LLM_ARCH_JINA_BERT_V2:
2571:         case LLM_ARCH_T5:
2572:         case LLM_ARCH_T5ENCODER:
2573:         case LLM_ARCH_JAIS:
2574:         case LLM_ARCH_RWKV6:
2575:         case LLM_ARCH_RWKV6QWEN2:
2576:         case LLM_ARCH_RWKV7:
2577:         case LLM_ARCH_ARWKV7:
2578:         case LLM_ARCH_WAVTOKENIZER_DEC:
2579:         case LLM_ARCH_NEMOTRON_H:
2580:         case LLM_ARCH_NEMOTRON_H_MOE:
2581:         case LLM_ARCH_KIMI_LINEAR:
2582:             return LLAMA_ROPE_TYPE_NONE;
2583: 
2584:         // use what we call a normal RoPE, operating on pairs of consecutive head values
2585:         case LLM_ARCH_LLAMA:
2586:         case LLM_ARCH_LLADA:
2587:         case LLM_ARCH_LLAMA4:
2588:         case LLM_ARCH_DECI:
2589:         case LLM_ARCH_BAICHUAN:
2590:         case LLM_ARCH_STARCODER:
2591:         case LLM_ARCH_INTERNLM2:
2592:         case LLM_ARCH_MINICPM:
2593:         case LLM_ARCH_XVERSE:
2594:         case LLM_ARCH_COMMAND_R:
2595:         case LLM_ARCH_COHERE2:
2596:         case LLM_ARCH_COHERE2MOE:
2597:         case LLM_ARCH_OLMO:
2598:         case LLM_ARCH_ARCTIC:
2599:         case LLM_ARCH_DEEPSEEK:
2600:         case LLM_ARCH_DEEPSEEK2:
2601:         case LLM_ARCH_DEEPSEEK2OCR:
2602:         case LLM_ARCH_DEEPSEEK32:
2603:         case LLM_ARCH_DEEPSEEK4:
2604:         case LLM_ARCH_MUSE_GLIMMER:
2605:         case LLM_ARCH_PLM:
2606:         case LLM_ARCH_CHATGLM:
2607:         case LLM_ARCH_GRANITE:
2608:         case LLM_ARCH_GRANITE_MOE:
2609:         case LLM_ARCH_GRANITE_HYBRID:
2610:         case LLM_ARCH_GRANITE_SWITCH:
2611:         case LLM_ARCH_CHAMELEON:
2612:         case LLM_ARCH_BAILINGMOE:
2613:         case LLM_ARCH_NEO_BERT:
2614:         case LLM_ARCH_SMOLLM3:
2615:         case LLM_ARCH_ARCEE:
2616:         case LLM_ARCH_ERNIE4_5:
2617:         case LLM_ARCH_ERNIE4_5_MOE:
2618:         case LLM_ARCH_MISTRAL3:
2619:         case LLM_ARCH_EAGLE3:
2620:         case LLM_ARCH_MISTRAL4:
2621:         case LLM_ARCH_LLAMA_EMBED:
2622:         case LLM_ARCH_MAINCODER:
2623:         case LLM_ARCH_GLM_DSA:
2624:         case LLM_ARCH_NANBEIGE:
2625:             return LLAMA_ROPE_TYPE_NORM;
2626: 
2627:         // the pairs of head values are offset by n_rot/2
2628:         case LLM_ARCH_FALCON:
2629:         case LLM_ARCH_FALCON_H1:
2630:         case LLM_ARCH_GROK:
2631:         case LLM_ARCH_DBRX:
2632:         case LLM_ARCH_BERT:
2633:         case LLM_ARCH_JINA_BERT_V3:
2634:         case LLM_ARCH_MODERN_BERT:
2635:         case LLM_ARCH_NOMIC_BERT:
2636:         case LLM_ARCH_NOMIC_BERT_MOE:
2637:         case LLM_ARCH_EUROBERT:
2638:         case LLM_ARCH_STABLELM:
2639:         case LLM_ARCH_BITNET:
2640:         case LLM_ARCH_QWEN:
2641:         case LLM_ARCH_QWEN2:
2642:         case LLM_ARCH_DREAM:
2643:         case LLM_ARCH_QWEN2MOE:
2644:         case LLM_ARCH_QWEN3:
2645:         case LLM_ARCH_QWEN3MOE:
2646:         case LLM_ARCH_LLADA_MOE:
2647:         case LLM_ARCH_RND1:
2648:         case LLM_ARCH_OLMO2:
2649:         case LLM_ARCH_OLMOE:
2650:         case LLM_ARCH_PHI2:
2651:         case LLM_ARCH_PHI3:
2652:         case LLM_ARCH_PHIMOE:
2653:         case LLM_ARCH_PLAMO:
2654:         case LLM_ARCH_PLAMO2:
2655:         case LLM_ARCH_PLAMO3:
2656:         case LLM_ARCH_GEMMA:
2657:         case LLM_ARCH_GEMMA2:
2658:         case LLM_ARCH_GEMMA3:
2659:         case LLM_ARCH_GEMMA3N:
2660:         case LLM_ARCH_GEMMA4:
2661:         case LLM_ARCH_GEMMA4_ASSISTANT:
2662:         case LLM_ARCH_GEMMA_EMBEDDING:
2663:         case LLM_ARCH_STARCODER2:
2664:         case LLM_ARCH_OPENELM:
2665:         case LLM_ARCH_GPTNEOX:
2666:         case LLM_ARCH_CODESHELL:
2667:         case LLM_ARCH_ORION:
2668:         case LLM_ARCH_NEMOTRON:
2669:         case LLM_ARCH_EXAONE:
2670:         case LLM_ARCH_EXAONE4:
2671:         case LLM_ARCH_EXAONE_MOE:
2672:         case LLM_ARCH_MINICPM3:
2673:         case LLM_ARCH_BAILINGMOE2:
2674:         case LLM_ARCH_DOTS1:
2675:         case LLM_ARCH_HUNYUAN_MOE:
2676:         case LLM_ARCH_JAIS2:
2677:         case LLM_ARCH_OPENAI_MOE:
2678:         case LLM_ARCH_HUNYUAN_DENSE:
2679:         case LLM_ARCH_HY_V3:
2680:         case LLM_ARCH_LFM2:
2681:         case LLM_ARCH_LFM2MOE:
2682:         case LLM_ARCH_SMALLTHINKER:
2683:         case LLM_ARCH_SEED_OSS:
2684:         case LLM_ARCH_GROVEMOE:
2685:         case LLM_ARCH_APERTUS:
2686:         case LLM_ARCH_MINIMAX_M2:
2687:         case LLM_ARCH_MINIMAX_M3:
2688:         case LLM_ARCH_COGVLM:
2689:         case LLM_ARCH_PANGU_EMBED:
2690:         case LLM_ARCH_AFMOE:
2691:         case LLM_ARCH_LAGUNA:
2692:         case LLM_ARCH_QWEN3NEXT:
2693:         case LLM_ARCH_MIMO2:
2694:         case LLM_ARCH_STEP35:
2695:         case LLM_ARCH_TALKIE:
2696:         case LLM_ARCH_MELLUM:
2697:             return LLAMA_ROPE_TYPE_NEOX;
2698: 
2699:         case LLM_ARCH_DFLASH:
2700:             // DSV4 DSpark drafters use DeepSeek-V4's normal RoPE; legacy DFlash backbones are NeoX
2701:             return model->hparams.dsv4_hc_mult > 0 ? LLAMA_ROPE_TYPE_NORM : LLAMA_ROPE_TYPE_NEOX;
2702: 
2703:         case LLM_ARCH_QWEN2VL:
2704:         case LLM_ARCH_PADDLEOCR:
2705:             return LLAMA_ROPE_TYPE_MROPE;
2706:         case LLM_ARCH_QWEN3VL:
2707:         case LLM_ARCH_QWEN3VLMOE:
2708:         case LLM_ARCH_QWEN35:
2709:         case LLM_ARCH_QWEN35MOE:
2710:         case LLM_ARCH_QWEN3TTS:
2711:             return LLAMA_ROPE_TYPE_IMROPE;
2712: 
2713:         case LLM_ARCH_GLM4:
2714:             return model->hparams.use_mrope() ? LLAMA_ROPE_TYPE_MROPE : LLAMA_ROPE_TYPE_NORM;
2715:         case LLM_ARCH_GLM4_MOE:
2716:             return model->hparams.use_mrope() ? LLAMA_ROPE_TYPE_MROPE : LLAMA_ROPE_TYPE_NEOX;
2717: 
2718:         case LLM_ARCH_HUNYUAN_VL:
2719:             return model->hparams.use_mrope() ? LLAMA_ROPE_TYPE_MROPE : LLAMA_ROPE_TYPE_NEOX;
2720: 
2721:         // all model arches should be listed explicitly here
2722:         case LLM_ARCH_UNKNOWN:
2723:             GGML_ABORT("unknown architecture");
2724:     }
2725: 
2726:     return LLAMA_ROPE_TYPE_NONE;
2727: }
```

**Key finding for your Rust port: `LLM_ARCH_QWEN35` and `LLM_ARCH_QWEN35MOE` map to `LLAMA_ROPE_TYPE_IMROPE`** (interleaved M-RoPE), NOT plain MROPE. QWEN2VL uses `LLAMA_ROPE_TYPE_MROPE`. The distinction is documented in the ggml.h API (section 3.3 below).

## 2.2 Rope-related hparams loading in `llama_model_base::load_hparams` (file 2)

`rope_sections` initialization (line 1144):
```cpp
1144:     std::fill(hparams.rope_sections.begin(), hparams.rope_sections.end(), 0);
```

Rope scaling metadata parsing (lines 1168–1194):
```cpp
1168:     bool rope_finetuned = false;
1169:     ml.get_key(LLM_KV_ROPE_SCALING_FINETUNED, rope_finetuned, false);
1170:     hparams.rope_finetuned = rope_finetuned;
1171: 
1172:     hparams.n_ctx_orig_yarn = hparams.n_ctx_train;
1173:     ml.get_key(LLM_KV_ROPE_SCALING_ORIG_CTX_LEN, hparams.n_ctx_orig_yarn, false);
1174: 
1175:     // rope_freq_base (optional)
1176:     hparams.rope_freq_base_train = 10000.0f;
1177:     ml.get_key(LLM_KV_ROPE_FREQ_BASE, hparams.rope_freq_base_train, false);
1178: 
1179:     std::string rope_scaling("linear");
1180:     ml.get_key(LLM_KV_ROPE_SCALING_TYPE, rope_scaling, false);
1181:     hparams.rope_scaling_type_train = llama_rope_scaling_type_from_string(rope_scaling);
1182:     GGML_ASSERT(hparams.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED);
1183: 
1184:     // TODO: Handle SWA metadata similarly when models start implementing it
1185:     // rope_freq_scale (inverse of the kv) is optional
1186:     float ropescale = 0.0f;
1187:     if (!ml.get_key(LLM_KV_ROPE_SCALING_FACTOR, ropescale, false)) {
1188:         // try the old key name
1189:         ml.get_key(LLM_KV_ROPE_SCALE_LINEAR, ropescale, false);
1190:     }
1191:     hparams.rope_freq_scale_train = ropescale == 0.0f ? 1.0f : 1.0f/ropescale;
1192: 
1193:     ml.get_key(LLM_KV_ROPE_SCALING_ATTN_FACTOR, hparams.rope_attn_factor, false);
1194:     ml.get_key(LLM_KV_ROPE_SCALING_ALPHA,       hparams.rope_scaling_alpha, false);
```

`n_rot` (i.e. `n_rot_full`) computation (lines 1196–1232):
```cpp
1196:     // non-transformer models do not have attention heads
1197:     if (hparams.n_head() > 0) {
1198:         // gpt-neox n_rot = rotary_pct * (n_embd / n_head)
1199:         // gpt-j n_rot = rotary_dim
1200: 
1201:         hparams.n_embd_head_k_full = hparams.n_embd / hparams.n_head();
1202:         ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH, hparams.n_embd_head_k_full, false);
1203: 
1204:         hparams.n_embd_head_v_full = hparams.n_embd / hparams.n_head();
1205:         ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH, hparams.n_embd_head_v_full, false);
1206: 
1207:         // sanity check for n_rot (optional)
1208:         hparams.n_rot_full = hparams.n_embd_head_k_full;
1209: 
1210:         ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT, hparams.n_rot_full, false);
1211: 
1212:         if (arch == LLM_ARCH_LLAMA || arch == LLM_ARCH_DECI || arch == LLM_ARCH_FALCON || arch == LLM_ARCH_LLAMA_EMBED) {
1213:             if (hparams.n_rot_full != hparams.n_embd_head_k_full) {
1214:                 throw std::runtime_error(format("invalid n_rot: %u, expected %u", hparams.n_rot_full, hparams.n_embd_head_k_full));
1215:             }
1216:         }
1217:     } else {
1218:         hparams.n_rot_full = 0;
1219:         hparams.n_embd_head_k_full = 0;
1220:         hparams.n_embd_head_v_full = 0;
1221:     }
1222: 
1223:     // head size and n_rot for SWA layers
1224:     {
1225:         hparams.n_embd_head_k_swa = hparams.n_embd_head_k_full;
1226:         hparams.n_embd_head_v_swa = hparams.n_embd_head_v_full;
1227:         ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA, hparams.n_embd_head_k_swa, false);
1228:         ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA, hparams.n_embd_head_v_swa, false);
1229: 
1230:         hparams.n_rot_swa = hparams.n_rot_full;
1231:         ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT_SWA, hparams.n_rot_swa, false);
1232:     }
```

And where `rope_type` gets set (line 1253):
```cpp
1253:     hparams.rope_type = llama_model_rope_type(this);
```

MRoPE sections logging (lines 1837–1840):
```cpp
1837:         // MRoPE (Multi-axis Rotary Position Embedding) sections
1838:         if (const auto & s = hparams.rope_sections; s[0] || s[1] || s[2] || s[3]) {
1839:             LLAMA_LOG_INFO("%s: mrope sections        = [%d, %d, %d, %d]\n", __func__, s[0], s[1], s[2], s[3]);
1840:         }
```

## 2.3 Items requested from file 2 that are NOT present

- **`build_norm`**: NOT in llama-model.cpp. It is a `llm_graph_context` method in `llama-graph.cpp` (see section 1.1 above).
- **`n_embd_s()`, `n_embd_r()`, `n_rot()`**: NOT defined in llama-model.cpp — these are `llama_hparams` methods (defined in the `llama-hparams.h` header, not among the two files). llama-model.cpp only *uses* `n_rot()` implicitly via `hparams.n_rot_full`/`n_rot_swa` (section 2.2) and via the graph context constructor `hparams.n_rot()` (file 1 line 1436). The usages of `hparams.n_embd_r()` and `hparams.n_embd_s()` appear in the graph code (file 1 line 3437/3460 and in the qwen3-5 patch lines 1604/1659/1694).
- **`llm_build_delta_net_base` / `build_conv_state`**: NOT present in llama-model.cpp. The delta-net code is named `llm_graph_context_delta::build_delta_net_unified*` and lives in the patch files (see Supplementary 3.2).

---

# SUPPLEMENTARY — critical rope/MoE reference that the two requested files do NOT contain

The actual `ggml_rope_ext` calls, the Qwen3.5/MoE layer builders, and the delta-net implementation live in sibling tool-output files in the same directory (the qwen3-5 patch `tool_fecf4a979001IEtaECugYEUziP` and the ggml.h header `tool_fe78b2add001vjKuzXB268jSvs`). These are verbatim from those files and are directly needed for your Rust port.

## 3.1 Qwen3.5 attention builder with the rope calls (patch file, lines 1399–1468)

```cpp
1399: +ggml_tensor * llm_build_qwen3_5::build_layer_attn(
1400: +        llm_graph_input_attn_kv * inp,
1401: +        ggml_tensor *             cur,
1402: +        ggml_tensor *             inp_pos,
1403: +        int                       il) {
1404: +    const int64_t n_embd_head = hparams.n_embd_head_v;
1405: +    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
1406: +
1407: +    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur); // [ (n_embd_head * 2) * n_head, n_tokens ]
1408: +    cb(Qcur_full, "Qcur_full", il);
1409: +
1410: +    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
1411: +        ggml_element_size(Qcur_full) * n_embd_head * 2,
1412: +        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
1413: +    cb(Qcur, "Qcur_reshaped", il);
1414: +
1415: +    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
1416: +    cb(Qcur, "Qcur_normed", il);
1417: +
1418: +    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
1419: +    cb(Kcur, "Kcur", il);
1420: +
1421: +    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
1422: +    cb(Vcur, "Vcur", il);
1423: +
1424: +    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
1425: +    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
1426: +    cb(Kcur, "Kcur_normed", il);
1427: +
1428: +    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
1429: +        ggml_element_size(Qcur_full) * n_embd_head * 2,
1430: +        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
1431: +        ggml_element_size(Qcur_full) * n_embd_head);
1432: +    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
1433: +    cb(gate, "gate_reshaped", il);
1434: +
1435: +    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
1436: +
1437: +    Qcur = ggml_rope_ext(
1438: +            ctx0, Qcur, inp_pos, nullptr,
1439: +            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
1440: +            ext_factor, attn_factor, beta_fast, beta_slow);
1441: +
1442: +    Kcur = ggml_rope_ext(
1443: +            ctx0, Kcur, inp_pos, nullptr,
1444: +            n_rot, rope_type, n_ctx_orig, freq_base,
1445: +            freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
1446: +
1447: +    cb(Qcur, "Qcur", il);
1448: +    cb(Kcur, "Kcur", il);
1449: +    cb(Vcur, "Vcur", il);
1450: +
1451: +    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;
1452: +
1453: +    cur = build_attn(inp,
1454: +                nullptr, nullptr,
1455: +                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
1456: +    cb(cur, "attn_pregate", il);
1457: +
1458: +    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
1459: +    cb(gate_sigmoid, "gate_sigmoid", il);
1460: +
1461: +    cur = ggml_mul(ctx0, cur, gate_sigmoid);
1462: +    cb(cur, "attn_gated", il);
1463: +
1464: +    cur = build_lora_mm(model.layers[il].wo, cur);
1465: +    cb(cur, "attn_output", il);
1466: +
1467: +    return cur;
1468: +}
```

**Key points:**
- `n_rot`, `rope_type`, `n_ctx_orig`, `freq_base`, `freq_scale`, `ext_factor`, `attn_factor`, `beta_fast`, `beta_slow` are all the `llm_graph_context` members from section 1.9. `rope_type` for QWEN35/QWEN35MOE is `LLAMA_ROPE_TYPE_IMROPE` (see section 2.1).
- Qwen3.5's `wq` outputs `(n_embd_head * 2) * n_head` — the second half is the attention gate, split off, sigmoided, and multiplied onto the attention output.
- Q and K are RMS-normed (`attn_q_norm`, `attn_k_norm`) BEFORE rope.
- `kq_scale = 1/sqrt(n_embd_head)` unless `hparams.f_attention_scale` is nonzero.
- Note: `ggml_rope_ext` is called here (IMROPE is passed as `mode`; for the actual MROPE `ggml_rope_multi` path with `sections`, see 3.3 — the interleaved/IMROPE and multi-axis layouts are handled by `ggml_rope_ext`/`ggml_rope_multi` based on `mode`).

## 3.2 Qwen3.5 MoE layer FFN — `llm_build_qwen3_5_moe::build_layer_ffn` (patch lines 1732–1781)

```cpp
1732: +llm_build_qwen3_5_moe::llm_build_qwen3_5_moe(const llama_model & model, const llm_graph_params & params) :
1733: +    llm_build_qwen3_5(model, params, defer_graph_build_t{}) {
1734: +    build_graph();
1735: +}
1736: +
1737: +ggml_tensor * llm_build_qwen3_5_moe::build_layer_ffn(ggml_tensor * cur, const int il) {
1738: +    // Check if this is an MoE layer
1739: +    if (model.layers[il].ffn_gate_inp != nullptr) {
1740: +        // MoE branch
1741: +        ggml_tensor * moe_out =
1742: +            build_moe_ffn(cur,
1743: +                model.layers[il].ffn_gate_inp, model.layers[il].ffn_up_exps,
1744: +                model.layers[il].ffn_gate_exps, model.layers[il].ffn_down_exps,
1745: +                nullptr,
1746: +                n_expert, n_expert_used, LLM_FFN_SILU,
1747: +                true, false, 0.0, LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il);
1748: +        cb(moe_out, "ffn_moe_out", il);
1749: +
1750: +        // Add shared experts if present
1751: +        if (model.layers[il].ffn_up_shexp != nullptr) {
1752: +            ggml_tensor * ffn_shexp =
1753: +                build_ffn(cur,
1754: +                    model.layers[il].ffn_up_shexp, NULL, NULL,
1755: +                    model.layers[il].ffn_gate_shexp, NULL, NULL,
1756: +                    model.layers[il].ffn_down_shexp, NULL, NULL,
1757: +                    NULL,
1758: +                    LLM_FFN_SILU, LLM_FFN_PAR, il);
1759: +            cb(ffn_shexp, "ffn_shexp", il);
1760: +
1761: +            // Apply shared expert gating (sigmoid)
1762: +            ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
1763: +            cb(shared_gate, "shared_expert_gate", il);
1764: +
1765: +            shared_gate = ggml_sigmoid(ctx0, shared_gate);
1766: +            cb(shared_gate, "shared_expert_gate_sigmoid", il);
1767: +
1768: +            ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
1769: +            cb(ffn_shexp, "ffn_shexp_gated", il);
1770: +
1771: +            cur = ggml_add(ctx0, moe_out, ffn_shexp);
1772: +            cb(cur, "ffn_out", il);
1773: +        } else {
1774: +            cur = moe_out;
1775: +        }
1776: +    } else {
1777: +        // Dense FFN branch (fallback)
1778: +        cur = llm_build_qwen3_5::build_layer_ffn(cur, il);
1779: +    }
1780: +    return cur;
1781: +}
```

**Key parameters for your Rust MoE port:** `n_expert`, `n_expert_used`, `type_op = LLM_FFN_SILU`, `norm_w = true`, `w_scale = 0.0`, `gating_op = LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX`, `il` layer index. Shared experts use `build_ffn` with `LLM_FFN_SILU, LLM_FFN_PAR` plus a separate `ffn_gate_inp_shexp` sigmoid gate.

The dense fallback (patch lines 1713–1723):
```cpp
1713: +ggml_tensor * llm_build_qwen3_5::build_layer_ffn(ggml_tensor * cur, const int il) {
1714: +    // Qwen3.5 Dense always uses dense FFN
1715: +    cur = build_ffn(cur,
1716: +        model.layers[il].ffn_up, NULL, NULL,
1717: +        model.layers[il].ffn_gate, NULL, NULL,
1718: +        model.layers[il].ffn_down, NULL, NULL,
1719: +        NULL,
1720: +        LLM_FFN_SILU, LLM_FFN_PAR, il);
1721: +    cb(cur, "ffn_out", il);
1722: +    return cur;
1723: +}
```

## 3.3 ggml.h rope API — mode constants and `ggml_rope_multi` (file `tool_fe78b2add001vjKuzXB268jSvs`)

Mode constants (lines 249–256):
```cpp
249: // TODO: convert to enum https://github.com/ggml-org/llama.cpp/pull/16187#discussion_r2388538726
250: #define GGML_ROPE_TYPE_NORMAL 0
251: #define GGML_ROPE_TYPE_NEOX   2
252: #define GGML_ROPE_TYPE_MROPE  8
253: #define GGML_ROPE_TYPE_VISION 24
254: #define GGML_ROPE_TYPE_IMROPE 40 // binary: 101000
255: 
256: #define GGML_MROPE_SECTIONS   4
```

`ggml_rope_multi` declaration with sections (lines 1839–1883):
```cpp
1839:     // multi-dimensional RoPE, for Qwen-VL and similar vision models
1840:     // mode can be either VISION, MROPE, IMROPE, cannot be combined with NORMAL or NEOX
1841:     // sections specify how many dimensions to rotate in each section:
1842:     //   section length is equivalent to number of cos/sin pairs, NOT the number of dims
1843:     //   (i.e. sum of 4 sections are expected to be n_dims/2)
1844:     //   last sections can be 0, means ignored
1845:     // all other options are identical to ggml_rope_ext
1846:     //
1847:     // important note:
1848:     //   - NEOX ordering is automatically applied and cannot be disabled for MROPE and VISION
1849:     //     if you need normal ordering, there are 2 methods:
1850:     //     (1) split the tensor manually using ggml_view
1851:     //     (2) permute the weight upon conversion
1852:     //   - for VISION, n_dims must be head_size/2
1853:     //
1854:     // example M-RoPE:
1855:     //  given sections = [t=4, y=2, x=2, 0]
1856:     //  given a single head with size = 18 --> [000000000000000000]
1857:     //  GGML_ROPE_TYPE_MROPE   n_dims = 16 --> [ttttyyxxttttyyxx00] (cos/sin are applied in NEOX ordering)
1858:     //  GGML_ROPE_TYPE_IMROPE  n_dims = 16 --> [ttyxttyxttyxttyx00] (interleaved M-RoPE, still NEOX ordering)
1859:     //  note: the theta for each dim is computed the same way as ggml_rope_ext, no matter the section
1860:     //        in other words, idx used for theta: [0123456789... until n_dims/2], not reset for each section
1861:     //
1862:     // example vision RoPE:
1863:     //  given sections = [y=4, x=4, 0, 0] (last 2 sections are ignored)
1864:     //  given a single head with size = 8 --> [00000000]
1865:     //  GGML_ROPE_TYPE_VISION  n_dims = 4 --> [yyyyxxxx]
1866:     //  other values of n_dims are untested and is undefined behavior
1867:     //  note: unlike MROPE, the theta for each dim is computed differently for each section
1868:     //        in other words, idx used for theta: [0123] for y section, then [0123] for x section
1869:     GGML_API struct ggml_tensor * ggml_rope_multi(
1870:             struct ggml_context * ctx,
1871:             struct ggml_tensor  * a,
1872:             struct ggml_tensor  * b,
1873:             struct ggml_tensor  * c,
1874:             int                   n_dims,
1875:             int                   sections[GGML_MROPE_SECTIONS],
1876:             int                   mode,
1877:             int                   n_ctx_orig,
1878:             float                 freq_base,
1879:             float                 freq_scale,
1880:             float                 ext_factor,
1881:             float                 attn_factor,
1882:             float                 beta_fast,
1883:             float                 beta_slow);
```

And `ggml_rope_ext` (lines 1824–1837):
```cpp
1824:     GGML_API struct ggml_tensor * ggml_rope_ext(
1825:             struct ggml_context * ctx,
1826:             struct ggml_tensor  * a,
1827:             struct ggml_tensor  * b,
1828:             struct ggml_tensor  * c,
1829:             int                   n_dims,
1830:             int                   mode,
1831:             int                   n_ctx_orig,
1832:             float                 freq_base,
1833:             float                 freq_scale,
1834:             float                 ext_factor,
1835:             float                 attn_factor,
1836:             float                 beta_fast,
1837:             float                 beta_slow);
```

## 3.4 Delta-net (recurrent linear attention) — `llm_graph_context_delta` (patch lines 596–1159, declarations 1178–1235)

The `build_layer_attn_linear` in the qwen3-5 builder (patch lines 1544–1711, already extracted in my search above) calls `build_delta_net_unified(ctx0, q_conv, k_conv, v_conv, gate, beta, state, causal_mask, identity, diag_mask, il, CHUNK_SIZE, hparams.f_norm_rms_eps)` at lines 1683–1685. The three delta-net functions (`build_delta_net_unified_chunking` at 596–979, `build_delta_net_unified_autoregressive` at 1002–1120, and the dispatcher `build_delta_net_unified` at 1134–1159) were extracted verbatim in my search output above (lines 560–1159). The class declaration in models.h (patch lines 1178–1235) is also captured above.

The `build_layer_attn_linear` body (lines 1544–1711) — including the conv-state handling (`build_rs`, `ggml_ssm_conv`, `state_update_target`, `n_embd_r()`/`n_embd_s()` usage) — is fully captured in my search output above. There is **no function named `llm_build_delta_net_base` or `build_conv_state`** anywhere in these files; the equivalent logic is `build_layer_attn_linear` + `build_delta_net_unified*` + `build_rs`.

---

## Summary of file paths

| Item | File | Lines |
|---|---|---|
| `build_ffn` | `/home/shinde/.local/share/opencode/tool-output/tool_fed18c9fc001qs0ktlg6o5wzyp` | 1669–1869 |
| `build_moe_ffn` (delegating) | same | 1871–1913 |
| `build_moe_ffn` (full) | same | 1915–2264 |
| `build_norm` | same | 1556–1589 |
| `build_lora_mm` / `build_lora_mm_id` | same | 1487–1516 / 1518–1554 |
| `build_qkv` | same | 1592–1666 |
| `build_attn_inp_kq_mask` | same | 28–45 |
| `build_attn_mha` | same | 2500–2633 |
| `build_attn` (7 overloads) | same | 2660–2710, 2745–2818, 2849–2905, 2907–2981, 2983–3068, 3070–3137, 3155–3196 |
| `llm_graph_context` ctor (`n_rot`, `rope_type`) | same | 1428–1471 |
| `build_rs` helpers | same | 3344–3378, 3410–3421 |
| `llama_model_rope_type` (QWEN35MOE → IMROPE) | `/home/shinde/.local/share/opencode/tool-output/tool_fed1805be0011zMzmwOdWpeTy3` | 2558–2727 |
| `n_rot_full`/`rope_sections` hparams loading | same | 1144, 1168–1232, 1253, 1837–1840 |
| Qwen3.5 `build_layer_attn` + `ggml_rope_ext` | `/home/shinde/.local/share/opencode/tool-output/tool_fecf4a979001IEtaECugYEUziP` | 1399–1468 |
| Qwen3.5 MoE layer FFN (build_moe_ffn call) | same | 1732–1781 |
| Qwen3.5 dense FFN | same | 1713–1723 |
| Qwen3.5 `build_layer_attn_linear` / delta-net | same | 1544–1711, 596–1159, 1178–1235 |
| `ggml_rope_multi` API + MROPE/IMROPE constants | `/home/shinde/.local/share/opencode/tool-output/tool_fe78b2add001vjKuzXB268jSvs` | 249–256, 1780–1915 |

Two notable caveats for your reimplementation: (1) the captured `llama-graph.cpp` is a post-refactor version where rope application is abstracted behind `self_k_rot`/`llama_mul_mat_hadamard`, and the explicit `ggml_rope_ext` calls for Qwen3.5 are in the `src/models/qwen3-5.cpp` patch hunks; (2) QWEN35/QWEN35MOE use `LLAMA_ROPE_TYPE_IMROPE` (interleaved M-RoPE, value 40), which the ggml docs say uses continuous theta indexing like `ggml_rope_ext` but applies cos/sin in interleaved M-RoPE layout under NEOX ordering.
</task_result>
</task>