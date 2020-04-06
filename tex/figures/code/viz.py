from .utils import *
from .geometry import get_angles
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Arc
import numpy as np


def visualize_simple(ax, b, theta, bo, ro, res=4999):

    # Find angles of intersection
    kappa, lam, xi, code = get_angles(b, theta, np.cos(theta), np.sin(theta), bo, ro)
    phi = kappa - np.pi / 2

    # HACK
    if (
        code == FLUX_NIGHT_VIS
        or code == FLUX_DAY_VIS
        or code == FLUX_QUAD_DAY_VIS
        or code == FLUX_QUAD_NIGHT_VIS
    ):
        phi = phi[::-1]

    color_night = (0, 0, 0, 0.75)
    color_day_occ = (0.314, 0.431, 0.514, 0.5)
    color_night_occ = (0, 0, 0, 0.5)

    # Equation of half-ellipse
    x = np.linspace(-1, 1, 10000)
    y = b * np.sqrt(1 - x ** 2)
    x_t = x * np.cos(theta) - y * np.sin(theta)
    y_t = x * np.sin(theta) + y * np.cos(theta)

    # Shaded regions
    p = np.linspace(-1, 1, res)
    xpt, ypt = np.meshgrid(p, p)
    cond1 = xpt ** 2 + (ypt - bo) ** 2 < ro ** 2  # inside occultor
    cond2 = xpt ** 2 + ypt ** 2 < 1  # inside occulted
    xr = xpt * np.cos(theta) + ypt * np.sin(theta)
    yr = -xpt * np.sin(theta) + ypt * np.cos(theta)
    cond3 = yr > b * np.sqrt(1 - xr ** 2)  # above terminator
    img_day_occ = np.zeros_like(xpt)
    img_day_occ[cond1 & cond2 & cond3] = 1
    img_night_occ = np.zeros_like(xpt)
    img_night_occ[cond1 & cond2 & ~cond3] = 1
    img_night = np.zeros_like(xpt)
    img_night[~cond1 & cond2 & ~cond3] = 1
    x0, y0 = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
    x = x0 * np.cos(theta) + y0 * np.sin(theta)
    y = -x0 * np.sin(theta) + y0 * np.cos(theta)
    z = np.sqrt(1 - x ** 2 - y ** 2)
    img_day = np.sqrt(1 - b ** 2) * y - b * z
    img_day[img_day < 0] = np.nan
    img_day[cond1] = np.nan

    # Draw basic shapes
    ax.axis("off")
    ax.add_artist(plt.Circle((0, bo), ro, fill=False, ls="--", lw=1.5))
    ax.add_artist(plt.Circle((0, 0), 1, fill=False))
    ax.plot(x_t, y_t, "k-", lw=1)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect(1)
    ax.imshow(
        img_day_occ,
        origin="lower",
        extent=(-1, 1, -1, 1),
        alpha=1,
        cmap=LinearSegmentedColormap.from_list(
            "cmap1", [(0, 0, 0, 0), color_day_occ], 2
        ),
        interpolation="none",
        zorder=-1,
    )
    ax.imshow(
        img_night_occ,
        origin="lower",
        extent=(-1, 1, -1, 1),
        alpha=1,
        cmap=LinearSegmentedColormap.from_list(
            "cmap1", [(0, 0, 0, 0), color_night_occ], 2
        ),
        interpolation="none",
        zorder=-1,
    )
    ax.imshow(
        img_day,
        origin="lower",
        extent=(-1, 1, -1, 1),
        alpha=1,
        cmap=LinearSegmentedColormap(
            "cmap1",
            segmentdata={
                "red": [[0.0, 0.0, 0.122], [1.0, 1.0, 1.0]],
                "green": [[0.0, 0.0, 0.467], [1.0, 1.0, 1.0]],
                "blue": [[0.0, 0.0, 0.706], [1.0, 1.0, 1.0]],
            },
            N=256,
        ),
        vmin=0,
        vmax=1,
        interpolation="none",
        zorder=-1,
    )
    ax.imshow(
        img_night,
        origin="lower",
        extent=(-1, 1, -1, 1),
        alpha=1,
        cmap=LinearSegmentedColormap.from_list("cmap1", [(0, 0, 0, 0), color_night], 2),
        interpolation="none",
        zorder=-1,
    )

    # Draw integration paths
    # T
    if len(xi):
        if (
            code == FLUX_NIGHT_VIS
            or code == FLUX_DAY_VIS
            or code == FLUX_QUAD_DAY_VIS
            or code == FLUX_QUAD_NIGHT_VIS
        ):
            inds = np.where(x_t ** 2 + (y_t - bo) ** 2 > ro ** 2)[0]
            if code == FLUX_QUAD_DAY_VIS or code == FLUX_QUAD_NIGHT_VIS:
                midpt = np.arange(len(x_t))[inds[1:]][np.diff(inds) > 1][0]
                x_t[midpt] = np.nan
        else:
            inds = np.where(x_t ** 2 + (y_t - bo) ** 2 < ro ** 2)[0]
            if code == FLUX_TRIP_DAY_OCC or code == FLUX_TRIP_NIGHT_OCC:
                midpt = np.arange(len(x_t))[inds[1:]][np.diff(inds) > 1][0]
                x_t[midpt] = np.nan
        ax.plot(x_t[inds], y_t[inds], "C1-", lw=2)

    if len(phi):
        for k in range(0, len(phi) // 2 + 1, 2):

            # P
            arc = Arc(
                (0, bo),
                2 * ro,
                2 * ro,
                0,
                phi[k] * 180 / np.pi,
                phi[k + 1] * 180 / np.pi,
                color="C1",
                lw=2,
                zorder=3,
            )
            ax.add_patch(arc)

    if len(lam):

        # Q
        arc = Arc(
            (0, 0),
            2,
            2,
            0,
            lam[0] * 180 / np.pi,
            lam[1] * 180 / np.pi,
            color="C1",
            lw=2,
            zorder=3,
        )
        ax.add_patch(arc)

    # Draw points of intersection
    for i, phi_i in enumerate(phi):

        # -- P --
        ax.plot(
            [ro * np.cos(phi_i)], [bo + ro * np.sin(phi_i)], "ko", ms=4, zorder=4,
        )

        # -- T --
        if len(xi):
            tol = 1e-7
            x_phi = ro * np.cos(phi_i)
            y_phi = bo + ro * np.sin(phi_i)
            x_xi = np.cos(theta) * np.cos(xi[i]) - b * np.sin(theta) * np.sin(xi[i])
            y_xi = np.sin(theta) * np.cos(xi[i]) + b * np.cos(theta) * np.sin(xi[i])
            if np.abs(y_phi - y_xi) < tol and np.abs(x_phi - x_xi) < tol:
                ax.plot(
                    [ro * np.cos(phi_i)],
                    [bo + ro * np.sin(phi_i)],
                    "ko",
                    ms=4,
                    zorder=4,
                )

    for i, lam_i in zip(range(len(lam)), lam):

        # -- Q --
        ax.plot(
            [np.cos(lam_i)], [np.sin(lam_i)], "ko", ms=4, zorder=4,
        )


def visualize(b, theta, bo, ro, res=4999):

    # Find angles of intersection
    kappa, lam, xi, code = get_angles(b, theta, np.cos(theta), np.sin(theta), bo, ro)
    phi = kappa - np.pi / 2

    color_night = (0, 0, 0, 0.75)
    color_day_occ = (0.314, 0.431, 0.514, 0.5)
    color_night_occ = (0, 0, 0, 0.5)

    # Equation of half-ellipse
    x = np.linspace(-1, 1, 1000)
    y = b * np.sqrt(1 - x ** 2)
    x_t = x * np.cos(theta) - y * np.sin(theta)
    y_t = x * np.sin(theta) + y * np.cos(theta)

    # Shaded regions
    p = np.linspace(-1, 1, res)
    xpt, ypt = np.meshgrid(p, p)
    cond1 = xpt ** 2 + (ypt - bo) ** 2 < ro ** 2  # inside occultor
    cond2 = xpt ** 2 + ypt ** 2 < 1  # inside occulted
    xr = xpt * np.cos(theta) + ypt * np.sin(theta)
    yr = -xpt * np.sin(theta) + ypt * np.cos(theta)
    cond3 = yr > b * np.sqrt(1 - xr ** 2)  # above terminator
    img_day_occ = np.zeros_like(xpt)
    img_day_occ[cond1 & cond2 & cond3] = 1
    img_night_occ = np.zeros_like(xpt)
    img_night_occ[cond1 & cond2 & ~cond3] = 1
    img_night = np.zeros_like(xpt)
    img_night[~cond1 & cond2 & ~cond3] = 1
    x0, y0 = np.meshgrid(np.linspace(-1, 1, res), np.linspace(-1, 1, res))
    x = x0 * np.cos(theta) + y0 * np.sin(theta)
    y = -x0 * np.sin(theta) + y0 * np.cos(theta)
    z = np.sqrt(1 - x ** 2 - y ** 2)
    img_day = np.sqrt(1 - b ** 2) * y - b * z
    img_day[img_day < 0] = np.nan
    img_day[cond1] = np.nan

    # Plot
    if len(lam):
        fig, ax = plt.subplots(1, 3, figsize=(14, 5))
        fig.subplots_adjust(left=0.025, right=0.975, bottom=0.05, top=0.825)
        ax[1].set_title(r"$T$", color="C1", fontsize=24)
        ax[0].set_title(r"$P$", color="C1", fontsize=24)
        ax[2].set_title(r"$Q$", color="C1", fontsize=24)
    else:
        fig, ax = plt.subplots(1, 2, figsize=(9, 5))
        fig.subplots_adjust(left=0.025, right=0.975, bottom=0.05, top=0.825)
        ax[1].set_title(r"$T$", color="C1", fontsize=24)
        ax[0].set_title(r"$P$", color="C1", fontsize=24)

    # Labels
    for i in range(len(phi)):
        ax[1].annotate(
            r"$\xi_{} = {:.1f}^\circ$".format(i, xi[i] * 180 / np.pi),
            xy=(0, 0),
            xycoords="axes fraction",
            xytext=(5, 25 - i * 20),
            textcoords="offset points",
            fontsize=10,
            color="k",
        )
        ax[0].annotate(
            r"$\phi_{} = {:.1f}^\circ$".format(i, phi[i] * 180 / np.pi),
            xy=(0, 0),
            xycoords="axes fraction",
            xytext=(5, 25 - i * 20),
            textcoords="offset points",
            fontsize=10,
            color="k",
        )
    for i in range(len(lam)):
        ax[2].annotate(
            r"$\lambda_{} = {:.1f}^\circ$".format(i, lam[i] * 180 / np.pi),
            xy=(0, 0),
            xycoords="axes fraction",
            xytext=(5, 25 - i * 20),
            textcoords="offset points",
            fontsize=10,
            color="k",
        )

    # Draw basic shapes
    for axis in ax:
        axis.axis("off")
        axis.add_artist(plt.Circle((0, bo), ro, lw=1.5, ls="--", fill=False))
        axis.add_artist(plt.Circle((0, 0), 1, fill=False))
        axis.plot(x_t, y_t, "k-", lw=1)
        axis.set_xlim(-1.25, 1.25)
        axis.set_ylim(-1.25, 1.25)
        axis.set_aspect(1)
        axis.imshow(
            img_day_occ,
            origin="lower",
            extent=(-1, 1, -1, 1),
            alpha=1,
            cmap=LinearSegmentedColormap.from_list(
                "cmap1", [(0, 0, 0, 0), color_day_occ], 2
            ),
        )
        axis.imshow(
            img_night_occ,
            origin="lower",
            extent=(-1, 1, -1, 1),
            alpha=1,
            cmap=LinearSegmentedColormap.from_list(
                "cmap1", [(0, 0, 0, 0), color_night_occ], 2
            ),
        )
        axis.imshow(
            img_day,
            origin="lower",
            extent=(-1, 1, -1, 1),
            alpha=1,
            cmap=LinearSegmentedColormap(
                "cmap1",
                segmentdata={
                    "red": [[0.0, 0.0, 0.122], [1.0, 1.0, 1.0]],
                    "green": [[0.0, 0.0, 0.467], [1.0, 1.0, 1.0]],
                    "blue": [[0.0, 0.0, 0.706], [1.0, 1.0, 1.0]],
                },
                N=256,
            ),
            vmin=0,
            vmax=1,
            interpolation="none",
            zorder=-1,
        )
        axis.imshow(
            img_night,
            origin="lower",
            extent=(-1, 1, -1, 1),
            alpha=1,
            cmap=LinearSegmentedColormap.from_list(
                "cmap1", [(0, 0, 0, 0), color_night], 2
            ),
        )

    # Draw integration paths

    # T
    inds = x_t ** 2 + (y_t - bo) ** 2 < ro ** 2
    ax[1].plot(x_t[inds], y_t[inds], "C1-", lw=3)
    ax[1].plot(
        [0, 1], [0, 0], color="k", ls="--", lw=0.5,
    )
    arc = Arc((0, 0), 0.15, 0.15, 0, 0, theta * 180 / np.pi, color="k", lw=0.5,)
    ax[1].add_patch(arc)
    ax[1].annotate(
        r"$\theta$",
        xy=(0.5 * 0.15 * np.cos(0.5 * theta), 0.5 * 0.15 * np.sin(0.5 * theta),),
        xycoords="data",
        xytext=(7 * np.cos(0.5 * theta), 7 * np.sin(0.5 * theta),),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=8,
        color="k",
    )

    if len(phi):
        for k in range(0, len(phi) // 2 + 1, 2):

            # P
            arc = Arc(
                (0, bo),
                2 * ro,
                2 * ro,
                0,
                phi[k] * 180 / np.pi,
                phi[k + 1] * 180 / np.pi,
                color="C1",
                lw=3,
                zorder=3,
            )
            ax[0].add_patch(arc)

    if len(lam):

        # Q
        arc = Arc(
            (0, 0),
            2,
            2,
            0,
            lam[0] * 180 / np.pi,
            lam[1] * 180 / np.pi,
            color="C1",
            lw=3,
            zorder=3,
        )
        ax[2].add_patch(arc)

    # Draw axes
    ax[1].plot(
        [-np.cos(theta), np.cos(theta)],
        [-np.sin(theta), np.sin(theta)],
        color="k",
        ls="--",
        lw=0.5,
    )
    ax[1].plot([0], [0], "ko", ms=4, zorder=4)
    ax[0].plot(
        [-ro, ro], [bo, bo], color="k", ls="--", lw=0.5,
    )
    ax[0].plot([0], [bo], "ko", ms=4, zorder=4)
    if len(lam):
        ax[2].plot(
            [-1, 1], [0, 0], color="k", ls="--", lw=0.5,
        )
        ax[2].plot(0, 0, "ko", ms=4, zorder=4)

    # Draw points of intersection & angles
    sz = [0.25, 0.5, 0.75, 1.0]
    for i, xi_i in enumerate(xi):

        # -- T --

        # xi angle
        ax[1].plot(
            [0, np.cos(np.sign(b) * xi_i + theta)],
            [0, np.sin(np.sign(b) * xi_i + theta)],
            color="k",
            lw=1,
        )

        # tangent line
        x0 = np.cos(xi_i) * np.cos(theta)
        y0 = np.cos(xi_i) * np.sin(theta)
        ax[1].plot(
            [x0, np.cos(np.sign(b) * xi_i + theta)],
            [y0, np.sin(np.sign(b) * xi_i + theta)],
            color="k",
            ls="--",
            lw=0.5,
        )

        # mark the polar angle
        ax[1].plot(
            [np.cos(np.sign(b) * xi_i + theta)],
            [np.sin(np.sign(b) * xi_i + theta)],
            "ko",
            ms=4,
            zorder=4,
        )

        # draw and label the angle arc
        if np.sin(xi_i) != 0:
            angle = sorted([theta, np.sign(b) * xi_i + theta])
            arc = Arc(
                (0, 0),
                sz[i],
                sz[i],
                0,
                angle[0] * 180 / np.pi,
                angle[1] * 180 / np.pi,
                color="k",
                lw=0.5,
            )
            ax[1].add_patch(arc)

            ax[1].annotate(
                r"$\xi_{}$".format(i),
                xy=(
                    0.5 * sz[i] * np.cos(0.5 * np.sign(b) * xi_i + theta),
                    0.5 * sz[i] * np.sin(0.5 * np.sign(b) * xi_i + theta),
                ),
                xycoords="data",
                xytext=(
                    7 * np.cos(0.5 * np.sign(b) * xi_i + theta),
                    7 * np.sin(0.5 * np.sign(b) * xi_i + theta),
                ),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8,
                color="k",
            )

        # points of intersection?
        tol = 1e-7
        for phi_i in phi:
            x_phi = ro * np.cos(phi_i)
            y_phi = bo + ro * np.sin(phi_i)
            x_xi = np.cos(theta) * np.cos(xi_i) - b * np.sin(theta) * np.sin(xi_i)
            y_xi = np.sin(theta) * np.cos(xi_i) + b * np.cos(theta) * np.sin(xi_i)
            if np.abs(y_phi - y_xi) < tol and np.abs(x_phi - x_xi) < tol:
                ax[1].plot(
                    [ro * np.cos(phi_i)],
                    [bo + ro * np.sin(phi_i)],
                    "ko",
                    ms=4,
                    zorder=4,
                )

    for i, phi_i in enumerate(phi):

        # -- P --

        # points of intersection
        ax[0].plot(
            [0, ro * np.cos(phi_i)],
            [bo, bo + ro * np.sin(phi_i)],
            color="k",
            ls="-",
            lw=1,
        )
        ax[0].plot(
            [ro * np.cos(phi_i)], [bo + ro * np.sin(phi_i)], "ko", ms=4, zorder=4,
        )

        # draw and label the angle arc
        angle = sorted([0, phi_i])
        arc = Arc(
            (0, bo),
            sz[i],
            sz[i],
            0,
            angle[0] * 180 / np.pi,
            angle[1] * 180 / np.pi,
            color="k",
            lw=0.5,
        )
        ax[0].add_patch(arc)
        ax[0].annotate(
            r"${}\phi_{}$".format("-" if phi_i < 0 else "", i),
            xy=(
                0.5 * sz[i] * np.cos(0.5 * (phi_i % (2 * np.pi))),
                bo + 0.5 * sz[i] * np.sin(0.5 * (phi_i % (2 * np.pi))),
            ),
            xycoords="data",
            xytext=(
                7 * np.cos(0.5 * (phi_i % (2 * np.pi))),
                7 * np.sin(0.5 * (phi_i % (2 * np.pi))),
            ),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8,
            color="k",
            zorder=4,
        )

    for i, lam_i in zip(range(len(lam)), lam):

        # -- Q --

        # points of intersection
        ax[2].plot(
            [0, np.cos(lam_i)], [0, np.sin(lam_i)], color="k", ls="-", lw=1,
        )
        ax[2].plot(
            [np.cos(lam_i)], [np.sin(lam_i)], "ko", ms=4, zorder=4,
        )

        # draw and label the angle arc
        angle = sorted([0, lam_i])
        arc = Arc(
            (0, 0),
            sz[i],
            sz[i],
            0,
            angle[0] * 180 / np.pi,
            angle[1] * 180 / np.pi,
            color="k",
            lw=0.5,
        )
        ax[2].add_patch(arc)
        ax[2].annotate(
            r"${}\lambda_{}$".format("-" if lam_i < 0 else "", i),
            xy=(0.5 * sz[i] * np.cos(0.5 * lam_i), 0.5 * sz[i] * np.sin(0.5 * lam_i),),
            xycoords="data",
            xytext=(7 * np.cos(0.5 * lam_i), 7 * np.sin(0.5 * lam_i),),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8,
            color="k",
            zorder=4,
        )

    return fig, ax
