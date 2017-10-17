import cPickle as pkl
import numpy as np
import matplotlib.pyplot as plt


def superpose(cdfs, outputpath, data):
    border_x = 0.5
    dx = 0.001
    x = np.arange(0, border_x, dx)
    plt.ioff()
    fig = plt.figure(figsize=(10,8))
    plt.xticks([0.01, 0.02, 0.05, 0.07, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    plt.xticks(rotation=70)
    plt.grid(b=True, which='major', axis='both', linestyle='dotted')
    floating = 3
    prec = "%." + str(floating) + "f"
    for cdf in cdfs:
        title = cdf["title"]
        auc = cdf["auc"]
        cdf01 = cdf["cdf01"]
        cdf_val = cdf["cdf"]
        plt.plot(x, cdf_val, marker=',',
                 label=title + ", CDF(0.1)=" + str(prec % (cdf01*100)) + "%, AUC=" +
                 str(prec % np.float(auc)) + "%")
    plt.legend(loc=4, prop={'size': 8}, fancybox=True, shadow=True)
    fig.suptitle('Cumulative distribution function (CDF) of NRMSE over ' + data + ' test set.')
    plt.xlabel('NRMSE')
    plt.ylabel('Data proportion')
    fig.savefig(outputpath, bbox_inches='tight', format='eps', dpi=1000)
    plt.ion()


def format_it(path, title):
    with open(path + "/errors.pkl", 'r') as f:
        stuff = pkl.load(f)
        auc = stuff["auc"]
        cdf = stuff["cdf"]
        cdf01 = stuff["cdf0_1"]
        model_cdf = {"auc": auc, "cdf": cdf, "cdf01": cdf01, "title": title}
        # mean shape
        auc_ms = stuff["auc_ms"]
        cdf_ms = stuff["cdf_ms"]
        cdf01_ms = stuff["cdf0_1_ms"]
        ms_cdf = {"auc": auc_ms, "cdf": cdf_ms, "cdf01": cdf01_ms,
                  "title": "CDF NRMSE mean shape"}
        return model_cdf, ms_cdf

if __name__ == "__main__":
    # LFPW, helen only labaled data.
    p_path = "../../exps/Reviewer1/Q2-SUM/"
    lfpw = {"mlp": p_path + "lfpw_10_08_2017_15_36_1507469809",
            "mlp+in": p_path + "lfpw_10_08_2017_23_18_1507497522",
            "mlp+out": p_path + "lfpw_10_08_2017_23_18_1507497524",
            "mlp+in+out": p_path + "lfpw_10_08_2017_23_18_1507497537"}
    helen = {"mlp": p_path + "helen_10_08_2017_15_36_1507469799",
             "mlp+in": p_path + "helen_10_08_2017_23_19_1507497550",
             "mlp+out": p_path + "helen_10_08_2017_23_19_1507497542",
             "mlp+in+out": p_path + "helen_10_08_2017_23_24_1507497848"}

    cdfs_lfpw, cdfs_helen = [], []
    # LFPW
    model_cdf, ms_cdf = format_it(lfpw["mlp"], "CDF NRMSE MLP")
    cdfs_lfpw.append(ms_cdf)
    cdfs_lfpw.append(model_cdf)

    model_cdf, _ = format_it(lfpw["mlp+in"], "CDF NRMSE MLP + in")
    cdfs_lfpw.append(model_cdf)
    model_cdf, _ = format_it(lfpw["mlp+out"], "CDF NRMSE MLP + out")
    cdfs_lfpw.append(model_cdf)
    model_cdf, _ = format_it(lfpw["mlp+in+out"], "CDF NRMSE MLP + in + out")
    cdfs_lfpw.append(model_cdf)
    superpose(cdfs_lfpw, p_path+"cdf_lfpw.eps", "LFPW")

    # HELEN
    model_cdf, ms_cdf = format_it(helen["mlp"], "CDF NRMSE MLP")
    cdfs_helen.append(ms_cdf)
    cdfs_helen.append(model_cdf)

    model_cdf, _ = format_it(helen["mlp+in"], "CDF NRMSE MLP + in")
    cdfs_helen.append(model_cdf)
    model_cdf, _ = format_it(helen["mlp+out"], "CDF NRMSE MLP + out")
    cdfs_helen.append(model_cdf)
    model_cdf, _ = format_it(helen["mlp+in+out"], "CDF NRMSE MLP + in + out")
    cdfs_helen.append(model_cdf)
    superpose(cdfs_helen, p_path+"cdf_helen.eps", "HELEN")
