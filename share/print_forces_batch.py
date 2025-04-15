from zutil.plot import Report
import matplotlib.pyplot as plt


def plot_report_batch(report: Report, variables: list = None, mean: int = 20):
    if not variables:
        # print available variables
        print("Available variables to plot:")
        for var in report.report.data.keys():
            if var not in report.report.residual_list + ["RealTimeStep", "Cycle"]:
                print(f"  {var}")

    else:
        # check if variable is valid:
        for var in variables:
            if var not in report.report.data.keys():
                raise ValueError(f"Variable '{var}' not found in report data.")
            else:
                print(f"Variable '{var}' is valid.  ")
                report.plot_forces(mean=mean)
                for cb in report.checkboxes:
                    if cb.description == var:
                        cb.value = True

                    report.plot_data(1)
                    fig = plt.gcf()
                    fig.savefig(f"plot_{var}.png")


plot_report_batch(r.reports[0])

plot_report_batch(r.reports[0], ["probe_cp"])
