from scipy import stats


def chi2_contingency(table):
    """
    卡方检验

    :param table: 是一个观测频数的二维数组
    :return 卡方统计量和p值
    """
    stat, p, dof, expected = stats.chi2_contingency(table)
    return stat, p


def fisher_exact(table):
    """
    Fisher精确检验
    只能用于2x2列联表

    :param table: 2x2列联表数组
    :return OR值和p值
    """
    odds_ratio, p = stats.fisher_exact(table)
    return odds_ratio, p


def t_test_ind(v1, v2):
    """
    独立样本t检验
    检验两个独立样本的均值是否有显著差异

    :param v1: 样本1的数组
    :param v2: 样本2的数组
    :return t统计量和p值
    """
    # 先检验方差齐性
    _, var_p = stats.levene(v1, v2)
    equal_var = var_p > 0.05
    stat, p = stats.ttest_ind(v1, v2, equal_var=equal_var)
    return stat, p


def t_test_rel(v1, v2):
    """
    配对或相关样本t检验
    检验两个相关样本的均值是否有显著差异

    :param v1: 样本1的数组
    :param v2: 样本2的数组
    :return t统计量和p值
    """
    stat, p = stats.ttest_rel(v1, v2)
    return stat, p


def oneway_anova(*args):
    """
    单因素方差分析
    检验两个或多个独立样本的均值是否有显著差异

    :param sample1, sample2, ... : array_like

    :return F统计量和p值
    """
    stat, p = stats.f_oneway(*args)
    return stat, p


def mannwhitneyu(v1, v2):
    """
    Mann-Whitney U 检验
    检验两个独立样本的分布是否相等
    独立样本t检验的非参数替代版本

    :param v1: 样本1的数组
    :param v2: 样本2的数组
    :return Mann-Whitney U 统计量和p值
    """
    stat, p = stats.mannwhitneyu(v1, v2)
    return stat, p


def wilcoxon(v1, v2):
    """
    Wilcoxon符号秩检验
    检验两个成对样本的分布是否相等
    配对样本t检验的非参数替代版本

    :param v1: 样本1的数组
    :param v2: 样本2的数组
    :return Wilcoxon统计量和p值
    """
    stat, p = stats.wilcoxon(v1, v2)
    return stat, p


def kruskal(v1, v2):
    """
    Kruskal-Wallis H 检验
    检验两个或多个独立样本的分布是否相等
    单因素方差分析的非参数替代版本

    :param v1: 样本1的数组
    :param v2: 样本2的数组
    :return Kruskal-Wallis H统计量和p值
    """
    stat, p = stats.kruskal(v1, v2)
    return stat, p


def friedmanchisquare(*args):
    """
    Friedman 检验
    检验两个或多个成对样本的分布是否相等，适用于重复测量资料
    Wilcoxon符号秩检验的多样本扩展

    :param v1: 样本1的数组
    :param v2: 样本2的数组
    :return Kruskal-Wallis H统计量和p值
    """
    stat, p = stats.friedmanchisquare(*args)
    return stat, p


def pearsonr(v1, v2):
    """
    皮尔逊相关系数和显著性检验。
    适用条件：连续变量，正态分布

    :param v1: 样本1的数组
    :param v2: 样本2的数组
    :return 皮尔逊相关系数和p值
    """
    r, p = stats.pearsonr(v1, v2)
    return r, p


def spearmanr(v1, v2):
    """
    斯皮尔曼秩相关系数。
    适用条件：可用于连续资料或等级资料，无正态分布假设

    :param v1: 样本1的数组
    :param v2: 样本2的数组
    :return 斯皮尔曼秩相关系数和p值
    """
    r, p = stats.spearmanr(v1, v2)
    return r, p


def kendalltau(v1, v2):
    """
    Kendall秩相关系数。
    衡量有序分类数据的序数相关性。相关系数1为极度相关、-1不极度不相关
    适用条件：有序分类变量，无正态分布假设

    :param v1: 样本1的数组
    :param v2: 样本2的数组
    :return Kendall tau相关系数和p值
    """
    r, p = stats.kendalltau(v1, v2)
    return r, p
