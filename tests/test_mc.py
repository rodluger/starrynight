def mc_search(seed=0, total=1000, epsrel=0.01, res=999):
    N = Numerical([1, 1, 1, 1], 0, 0, 0, 0)
    np.random.seed(seed)
    for i in tqdm(range(total)):
        N.bo = 0
        N.ro = 2
        while (N.bo <= N.ro - 1) or (N.bo >= 1 + N.ro):
            if np.random.random() > 0.5:
                N.ro = np.random.random() * 10
                N.bo = np.random.random() * 20
            else:
                N.ro = np.random.random()
                N.bo = np.random.random() * 2
        N.theta = np.random.random() * 2 * np.pi
        N.b = 1 - 2 * np.random.random()

        flux = N.flux()
        flux_brute = N.flux_brute(res=res)
        if np.abs(flux - flux_brute) > epsrel:
            N.visualize(name="{:04d}".format(i), res=res)
