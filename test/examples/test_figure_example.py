import os
import sys

path_file = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path_file + "/../../examples/")

import pytest


def seaborn_installed():
    try:
        import seaborn

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not seaborn_installed(), reason="seaborn is not installed")
@pytest.mark.mpl_image_compare(
    style="default",
    baseline_dir=path_file + "/baseline",
    filename="plot_2D_simulation_example_1.png",
)
@pytest.mark.example
def test_plot_2D_simulation_example():
    import matplotlib.pyplot as plt
    import plot_2D_simulation_example

    return plt.figure(num=1)


@pytest.mark.skipif(not seaborn_installed(), reason="seaborn is not installed")
@pytest.mark.mpl_image_compare(
    style="default",
    baseline_dir=path_file + "/baseline",
    filename="plot_conditional_vs_marginal_xor_data_1.png",
)
@pytest.mark.example
def test_plot_conditional_vs_marginal_xor_data_1():
    import matplotlib.pyplot as plt
    import plot_conditional_vs_marginal_xor_data

    return plt.figure(num=1)


@pytest.mark.skipif(not seaborn_installed(), reason="seaborn is not installed")
@pytest.mark.mpl_image_compare(
    style="default",
    baseline_dir=path_file + "/baseline",
    filename="plot_conditional_vs_marginal_xor_data_2.png",
)
@pytest.mark.example
def test_plot_conditional_vs_marginal_xor_data_2():
    import matplotlib.pyplot as plt
    import plot_conditional_vs_marginal_xor_data

    return plt.figure(num=2)


@pytest.mark.skipif(not seaborn_installed(), reason="seaborn is not installed")
@pytest.mark.mpl_image_compare(
    style="default",
    baseline_dir=path_file + "/baseline",
    filename="plot_dcrt_example_1.png",
)
@pytest.mark.example
def test_plot_dcrt_example():
    import matplotlib.pyplot as plt
    import plot_dcrt_example

    return plt.figure(num=1)


@pytest.mark.skipif(not seaborn_installed(), reason="seaborn is not installed")
@pytest.mark.mpl_image_compare(
    style="default",
    baseline_dir=path_file + "/baseline",
    filename="plot_diabetes_variable_importance_example_1.png",
)
@pytest.mark.example
def test_plot_diabetes_variable_importance_example():
    import matplotlib.pyplot as plt
    import plot_diabetes_variable_importance_example

    return plt.figure(num=1)


@pytest.mark.skipif(not seaborn_installed(), reason="seaborn is not installed")
@pytest.mark.mpl_image_compare(
    style="default",
    baseline_dir=path_file + "/baseline",
    filename="plot_fmri_data_example_1.png",
)
@pytest.mark.example
def test_plot_fmri_data_example_1():
    import matplotlib.pyplot as plt
    import plot_fmri_data_example

    return plt.figure(num=1)


@pytest.mark.skipif(not seaborn_installed(), reason="seaborn is not installed")
@pytest.mark.mpl_image_compare(
    style="default",
    baseline_dir=path_file + "/baseline",
    filename="plot_fmri_data_example_2.png",
)
@pytest.mark.example
def test_plot_fmri_data_example_2():
    import matplotlib.pyplot as plt
    import plot_fmri_data_example

    return plt.figure(num=2)


@pytest.mark.skipif(not seaborn_installed(), reason="seaborn is not installed")
@pytest.mark.mpl_image_compare(
    style="default",
    baseline_dir=path_file + "/baseline",
    filename="plot_importance_classification_iris_1.png",
)
@pytest.mark.example
def test_plot_importance_classification_iris_1():
    import matplotlib.pyplot as plt
    import plot_importance_classification_iris

    return plt.figure(num=1)


@pytest.mark.skipif(not seaborn_installed(), reason="seaborn is not installed")
@pytest.mark.mpl_image_compare(
    style="default",
    baseline_dir=path_file + "/baseline",
    filename="plot_knockoff_aggregation_1.png",
)
@pytest.mark.example
def test_plot_knockoff_aggregation_1():
    import matplotlib.pyplot as plt
    import plot_knockoff_aggregation

    return plt.figure(num=1)


@pytest.mark.skipif(not seaborn_installed(), reason="seaborn is not installed")
@pytest.mark.mpl_image_compare(
    style="default",
    baseline_dir=path_file + "/baseline",
    filename="plot_knockoff_aggregation_2.png",
)
@pytest.mark.example
def test_plot_knockoff_aggregation_2():
    import matplotlib.pyplot as plt
    import plot_knockoff_aggregation

    return plt.figure(num=2)


@pytest.mark.skipif(not seaborn_installed(), reason="seaborn is not installed")
@pytest.mark.mpl_image_compare(
    style="default",
    baseline_dir=path_file + "/baseline",
    filename="plot_knockoff_aggregation_3.png",
)
@pytest.mark.example
def test_plot_knockoff_aggregation_3():
    import matplotlib.pyplot as plt
    import plot_knockoff_aggregation

    return plt.figure(num=3)


@pytest.mark.skipif(not seaborn_installed(), reason="seaborn is not installed")
@pytest.mark.mpl_image_compare(
    style="default",
    baseline_dir=path_file + "/baseline",
    filename="plot_knockoff_aggregation_4.png",
)
@pytest.mark.example
def test_plot_knockoff_aggregation_4():
    import matplotlib.pyplot as plt
    import plot_knockoff_aggregation

    return plt.figure(num=4)


@pytest.mark.skipif(not seaborn_installed(), reason="seaborn is not installed")
@pytest.mark.mpl_image_compare(
    style="default",
    baseline_dir=path_file + "/baseline",
    filename="plot_knockoff_wisconsin_1.png",
)
@pytest.mark.example
def test_plot_knockoff_wisconsin():
    import matplotlib.pyplot as plt
    import plot_knockoffs_wisconsin

    return plt.figure(num=1)


@pytest.mark.skipif(not seaborn_installed(), reason="seaborn is not installed")
@pytest.mark.mpl_image_compare(
    style="default",
    baseline_dir=path_file + "/baseline",
    filename="plot_model_agnostic_importance_1.png",
)
@pytest.mark.example
def test_plot_model_agnostic_importance_1():
    import matplotlib.pyplot as plt
    import plot_model_agnostic_importance

    return plt.figure(num=1)


@pytest.mark.skipif(not seaborn_installed(), reason="seaborn is not installed")
@pytest.mark.mpl_image_compare(
    style="default",
    baseline_dir=path_file + "/baseline",
    filename="plot_model_agnostic_importance_2.png",
)
@pytest.mark.example
def test_plot_model_agnostic_importance_2():
    import matplotlib.pyplot as plt
    import plot_model_agnostic_importance

    return plt.figure(num=2)


@pytest.mark.skipif(not seaborn_installed(), reason="seaborn is not installed")
@pytest.mark.mpl_image_compare(
    style="default",
    baseline_dir=path_file + "/baseline",
    filename="plot_pitfalls_permutation_importance_1.png",
)
@pytest.mark.example
def test_plot_pitfalls_permutation_importance_1():
    import matplotlib.pyplot as plt
    import plot_pitfalls_permutation_importance

    return plt.figure(num=1)


@pytest.mark.skipif(not seaborn_installed(), reason="seaborn is not installed")
@pytest.mark.mpl_image_compare(
    style="default",
    baseline_dir=path_file + "/baseline",
    filename="plot_pitfalls_permutation_importance_2.png",
)
@pytest.mark.example
def test_plot_pitfalls_permutation_importance_2():
    import matplotlib.pyplot as plt
    import plot_pitfalls_permutation_importance

    return plt.figure(num=2)


@pytest.mark.skipif(not seaborn_installed(), reason="seaborn is not installed")
@pytest.mark.mpl_image_compare(
    style="default",
    baseline_dir=path_file + "/baseline",
    filename="plot_pitfalls_permutation_importance_3.png",
)
@pytest.mark.example
def test_plot_pitfalls_permutation_importance_3():
    import matplotlib.pyplot as plt
    import plot_pitfalls_permutation_importance

    return plt.figure(num=3)


@pytest.mark.skipif(not seaborn_installed(), reason="seaborn is not installed")
@pytest.mark.mpl_image_compare(
    style="default",
    baseline_dir=path_file + "/baseline",
    filename="plot_pitfalls_permutation_importance_4.png",
)
@pytest.mark.example
def test_plot_pitfalls_permutation_importance_4():
    import matplotlib.pyplot as plt
    import plot_pitfalls_permutation_importance

    return plt.figure(num=4)
