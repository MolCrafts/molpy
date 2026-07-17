window.BENCHMARK_DATA = {
  "lastUpdate": 1784273122773,
  "repoUrl": "https://github.com/MolCrafts/molpy",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "42854324+Roy-Kid@users.noreply.github.com",
            "name": "Jichen Li",
            "username": "Roy-Kid"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7fa1b1a333d368d761547132af4bf67417f4acf2",
          "message": "Merge pull request #35 from Roy-Kid/dev\n\nchore(release): molpy 0.5.0",
          "timestamp": "2026-06-21T13:54:49+02:00",
          "tree_id": "0c6f346e45e4c72a04a1889c82c2b6217fe1a685",
          "url": "https://github.com/MolCrafts/molpy/commit/7fa1b1a333d368d761547132af4bf67417f4acf2"
        },
        "date": 1782043509114,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 14125.332030682193,
            "unit": "iter/sec",
            "range": "stddev: 0.000009140194603872883",
            "extra": "mean: 70.79479603225329 usec\nrounds: 1613"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[small-1k]",
            "value": 15795.454685438599,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033431982059597546",
            "extra": "mean: 63.30935195691915 usec\nrounds: 10399"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[medium-10k]",
            "value": 1599.9648955237533,
            "unit": "iter/sec",
            "range": "stddev: 0.000011449258531703663",
            "extra": "mean: 625.0137129869009 usec\nrounds: 1540"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[large-100k]",
            "value": 156.4756068049338,
            "unit": "iter/sec",
            "range": "stddev: 0.0002989298855934115",
            "extra": "mean: 6.390772468750505 msec\nrounds: 160"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[small-1k]",
            "value": 15906.577384502665,
            "unit": "iter/sec",
            "range": "stddev: 0.0000071802384311566186",
            "extra": "mean: 62.867075413361526 usec\nrounds: 14878"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[medium-10k]",
            "value": 1581.1057063739322,
            "unit": "iter/sec",
            "range": "stddev: 0.0000852554840062686",
            "extra": "mean: 632.4687817953517 usec\nrounds: 1604"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[large-100k]",
            "value": 161.51179774300542,
            "unit": "iter/sec",
            "range": "stddev: 0.00004096283599088842",
            "extra": "mean: 6.191498169014139 msec\nrounds: 142"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[small-1k]",
            "value": 4763.302759070893,
            "unit": "iter/sec",
            "range": "stddev: 0.000013251743278348423",
            "extra": "mean: 209.93836641932353 usec\nrounds: 3234"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[medium-10k]",
            "value": 636.011210899167,
            "unit": "iter/sec",
            "range": "stddev: 0.00002341364369716529",
            "extra": "mean: 1.5722993287905103 msec\nrounds: 587"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[large-100k]",
            "value": 64.06788465464768,
            "unit": "iter/sec",
            "range": "stddev: 0.0008091828443549018",
            "extra": "mean: 15.60844415872964 msec\nrounds: 63"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[small-1k]",
            "value": 74330.82282116033,
            "unit": "iter/sec",
            "range": "stddev: 0.000002572287923846006",
            "extra": "mean: 13.453369168346166 usec\nrounds: 8874"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[medium-10k]",
            "value": 64315.95006065362,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030956279978153665",
            "extra": "mean: 15.5482426840767 usec\nrounds: 23442"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[large-100k]",
            "value": 26321.07155347829,
            "unit": "iter/sec",
            "range": "stddev: 0.0000037791568697537372",
            "extra": "mean: 37.992374207419054 usec\nrounds: 8832"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[small-1k]",
            "value": 326401.50054763485,
            "unit": "iter/sec",
            "range": "stddev: 7.553077493983962e-7",
            "extra": "mean: 3.0637114054996832 usec\nrounds: 34124"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[medium-10k]",
            "value": 328036.3911470775,
            "unit": "iter/sec",
            "range": "stddev: 7.741294606859576e-7",
            "extra": "mean: 3.0484422673447926 usec\nrounds: 92937"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[large-100k]",
            "value": 329983.0384410641,
            "unit": "iter/sec",
            "range": "stddev: 7.101573578324681e-7",
            "extra": "mean: 3.0304587918345467 usec\nrounds: 88332"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[small-1k]",
            "value": 40.511408331023915,
            "unit": "iter/sec",
            "range": "stddev: 0.0054810294915871765",
            "extra": "mean: 24.68440474418642 msec\nrounds: 43"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[medium-10k]",
            "value": 3.984745727107178,
            "unit": "iter/sec",
            "range": "stddev: 0.002474213560109309",
            "extra": "mean: 250.9570417999981 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[large-50k]",
            "value": 0.7331765250769496,
            "unit": "iter/sec",
            "range": "stddev: 0.03305632665465463",
            "extra": "mean: 1.363928011599998 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[small-1k]",
            "value": 363.3546614931862,
            "unit": "iter/sec",
            "range": "stddev: 0.0026777008462808226",
            "extra": "mean: 2.752132024096112 msec\nrounds: 415"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[medium-10k]",
            "value": 31.198895088634803,
            "unit": "iter/sec",
            "range": "stddev: 0.00937434993606822",
            "extra": "mean: 32.05241714999971 msec\nrounds: 40"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[large-50k]",
            "value": 5.508987177523841,
            "unit": "iter/sec",
            "range": "stddev: 0.012289858919012266",
            "extra": "mean: 181.5215697142857 msec\nrounds: 7"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "42854324+Roy-Kid@users.noreply.github.com",
            "name": "Jichen Li",
            "username": "Roy-Kid"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9e6081663ad0d302f4cf614776405dca6cae9a1e",
          "message": "Merge pull request #36 from MolCrafts/dev\n\nchore(release): molpy 0.5.1",
          "timestamp": "2026-07-01T17:21:22+08:00",
          "tree_id": "cdb397714c96fa4febd6a55c2b8bf3c345f8eb8f",
          "url": "https://github.com/MolCrafts/molpy/commit/9e6081663ad0d302f4cf614776405dca6cae9a1e"
        },
        "date": 1782897740778,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 16218.991634563965,
            "unit": "iter/sec",
            "range": "stddev: 0.000007485567760484612",
            "extra": "mean: 61.65611417351744 usec\nrounds: 1524"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[small-1k]",
            "value": 16976.529797659656,
            "unit": "iter/sec",
            "range": "stddev: 0.0000029810633019183284",
            "extra": "mean: 58.90485346056163 usec\nrounds: 10591"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[medium-10k]",
            "value": 1709.8066468390737,
            "unit": "iter/sec",
            "range": "stddev: 0.000009892181954724389",
            "extra": "mean: 584.8614531056502 usec\nrounds: 1642"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[large-100k]",
            "value": 169.2868611529697,
            "unit": "iter/sec",
            "range": "stddev: 0.00006329770208711233",
            "extra": "mean: 5.907132976471148 msec\nrounds: 170"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[small-1k]",
            "value": 17104.285847048126,
            "unit": "iter/sec",
            "range": "stddev: 0.000003306082165845423",
            "extra": "mean: 58.464878857984054 usec\nrounds: 15552"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[medium-10k]",
            "value": 1711.5289330818703,
            "unit": "iter/sec",
            "range": "stddev: 0.00001815923677335843",
            "extra": "mean: 584.2729156785838 usec\nrounds: 1518"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[large-100k]",
            "value": 172.0222747641101,
            "unit": "iter/sec",
            "range": "stddev: 0.00003670019615746907",
            "extra": "mean: 5.813200653062374 msec\nrounds: 147"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[small-1k]",
            "value": 5219.283440499827,
            "unit": "iter/sec",
            "range": "stddev: 0.000016773392289998205",
            "extra": "mean: 191.5971821419675 usec\nrounds: 3371"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[medium-10k]",
            "value": 652.6767156674407,
            "unit": "iter/sec",
            "range": "stddev: 0.00008636656076482775",
            "extra": "mean: 1.5321520991864699 msec\nrounds: 615"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[large-100k]",
            "value": 67.90993103757282,
            "unit": "iter/sec",
            "range": "stddev: 0.00006397902440499643",
            "extra": "mean: 14.72538676923005 msec\nrounds: 65"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[small-1k]",
            "value": 74277.82729134634,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023845269862052285",
            "extra": "mean: 13.46296783934745 usec\nrounds: 8271"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[medium-10k]",
            "value": 61081.12317192248,
            "unit": "iter/sec",
            "range": "stddev: 0.00000568922142874905",
            "extra": "mean: 16.37167013424658 usec\nrounds: 22206"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[large-100k]",
            "value": 30000.068233033082,
            "unit": "iter/sec",
            "range": "stddev: 0.000003545810365101284",
            "extra": "mean: 33.33325751902456 usec\nrounds: 9110"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[small-1k]",
            "value": 331817.6674423225,
            "unit": "iter/sec",
            "range": "stddev: 7.71639808677959e-7",
            "extra": "mean: 3.0137033019009545 usec\nrounds: 19262"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[medium-10k]",
            "value": 331874.8969621636,
            "unit": "iter/sec",
            "range": "stddev: 7.446419825057571e-7",
            "extra": "mean: 3.013183609708233 usec\nrounds: 89358"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[large-100k]",
            "value": 331548.2326301037,
            "unit": "iter/sec",
            "range": "stddev: 7.739809165122711e-7",
            "extra": "mean: 3.016152407350226 usec\nrounds: 93985"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[small-1k]",
            "value": 39.29810240899715,
            "unit": "iter/sec",
            "range": "stddev: 0.006500037212165738",
            "extra": "mean: 25.44652129999676 msec\nrounds: 40"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[medium-10k]",
            "value": 3.8823175233874414,
            "unit": "iter/sec",
            "range": "stddev: 0.010737447881435467",
            "extra": "mean: 257.57810740000195 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[large-50k]",
            "value": 0.7181288781296088,
            "unit": "iter/sec",
            "range": "stddev: 0.06434563868927785",
            "extra": "mean: 1.392507710599989 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[small-1k]",
            "value": 396.3811027534655,
            "unit": "iter/sec",
            "range": "stddev: 0.0017266284464410567",
            "extra": "mean: 2.5228246075645115 msec\nrounds: 423"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[medium-10k]",
            "value": 31.44815292164672,
            "unit": "iter/sec",
            "range": "stddev: 0.00827979018511335",
            "extra": "mean: 31.798369923076457 msec\nrounds: 39"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[large-50k]",
            "value": 5.84646602015496,
            "unit": "iter/sec",
            "range": "stddev: 0.014508693113656228",
            "extra": "mean: 171.04349816669165 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "42854324+Roy-Kid@users.noreply.github.com",
            "name": "Jichen Li",
            "username": "Roy-Kid"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "09bd84cdb4667371af74da0fdfe2cd3e332e215a",
          "message": "Merge pull request #37 from MolCrafts/ci/pytest-xdist\n\nci: parallelize test suite with pytest-xdist (-n auto)",
          "timestamp": "2026-07-01T22:30:47+08:00",
          "tree_id": "fb33a35677904993190f1e8fe095aa461f710066",
          "url": "https://github.com/MolCrafts/molpy/commit/09bd84cdb4667371af74da0fdfe2cd3e332e215a"
        },
        "date": 1782916299721,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 20844.03698806268,
            "unit": "iter/sec",
            "range": "stddev: 0.000003459802719402555",
            "extra": "mean: 47.97535144332632 usec\nrounds: 1767"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[small-1k]",
            "value": 18089.210742122625,
            "unit": "iter/sec",
            "range": "stddev: 0.000002552900468142631",
            "extra": "mean: 55.281571664782206 usec\nrounds: 11491"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[medium-10k]",
            "value": 1896.4584758686265,
            "unit": "iter/sec",
            "range": "stddev: 0.000011891625174009932",
            "extra": "mean: 527.2986531075901 usec\nrounds: 1770"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[large-100k]",
            "value": 185.89444925967854,
            "unit": "iter/sec",
            "range": "stddev: 0.00003629544653842702",
            "extra": "mean: 5.379396770492518 msec\nrounds: 183"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[small-1k]",
            "value": 18166.003092775667,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024852387122714463",
            "extra": "mean: 55.047882293804314 usec\nrounds: 15887"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[medium-10k]",
            "value": 1865.8591806686068,
            "unit": "iter/sec",
            "range": "stddev: 0.000009884446009238721",
            "extra": "mean: 535.9461262460668 usec\nrounds: 1806"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[large-100k]",
            "value": 182.86220959738935,
            "unit": "iter/sec",
            "range": "stddev: 0.0001135008360717369",
            "extra": "mean: 5.468598472050163 msec\nrounds: 161"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[small-1k]",
            "value": 6050.188043112055,
            "unit": "iter/sec",
            "range": "stddev: 0.000004792886479843782",
            "extra": "mean: 165.2841189189926 usec\nrounds: 3885"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[medium-10k]",
            "value": 722.4529392458163,
            "unit": "iter/sec",
            "range": "stddev: 0.000018090903338658394",
            "extra": "mean: 1.3841732044774027 msec\nrounds: 670"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[large-100k]",
            "value": 73.60304813524077,
            "unit": "iter/sec",
            "range": "stddev: 0.00006961253835139757",
            "extra": "mean: 13.586393842855061 msec\nrounds: 70"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[small-1k]",
            "value": 80550.98809929074,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013083778586788164",
            "extra": "mean: 12.414496998688028 usec\nrounds: 9829"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[medium-10k]",
            "value": 72543.41042094892,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010352989770796686",
            "extra": "mean: 13.784849570722999 usec\nrounds: 20621"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[large-100k]",
            "value": 26227.479773416013,
            "unit": "iter/sec",
            "range": "stddev: 0.000002679558369300938",
            "extra": "mean: 38.127948572992246 usec\nrounds: 7078"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[small-1k]",
            "value": 353021.42749588756,
            "unit": "iter/sec",
            "range": "stddev: 3.5839628453409e-7",
            "extra": "mean: 2.832689242387841 usec\nrounds: 39098"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[medium-10k]",
            "value": 353741.63026215695,
            "unit": "iter/sec",
            "range": "stddev: 3.4629090899351504e-7",
            "extra": "mean: 2.826922008752272 usec\nrounds: 108743"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[large-100k]",
            "value": 352512.47711111925,
            "unit": "iter/sec",
            "range": "stddev: 3.7871041647520164e-7",
            "extra": "mean: 2.8367790218238977 usec\nrounds: 88692"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[small-1k]",
            "value": 44.871700604073006,
            "unit": "iter/sec",
            "range": "stddev: 0.004828985021163893",
            "extra": "mean: 22.285761104165285 msec\nrounds: 48"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[medium-10k]",
            "value": 4.329733024946742,
            "unit": "iter/sec",
            "range": "stddev: 0.009696357982599386",
            "extra": "mean: 230.96112260000154 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[large-50k]",
            "value": 0.7916267473046825,
            "unit": "iter/sec",
            "range": "stddev: 0.02496302656775112",
            "extra": "mean: 1.2632215919999965 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[small-1k]",
            "value": 423.3344716210705,
            "unit": "iter/sec",
            "range": "stddev: 0.0021426424366470607",
            "extra": "mean: 2.3621983727682507 msec\nrounds: 448"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[medium-10k]",
            "value": 37.371306714867394,
            "unit": "iter/sec",
            "range": "stddev: 0.007225301073194106",
            "extra": "mean: 26.75849703703753 msec\nrounds: 27"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[large-50k]",
            "value": 6.469077026283638,
            "unit": "iter/sec",
            "range": "stddev: 0.01316304769707417",
            "extra": "mean: 154.58155714285584 msec\nrounds: 7"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "42854324+Roy-Kid@users.noreply.github.com",
            "name": "Jichen Li",
            "username": "Roy-Kid"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "00978b0736346192ecf68c71f7ce8b6b63a01c14",
          "message": "Merge pull request #39 from MolCrafts/ci/tests-data-skip-con\n\nci: skip the con/ dir via sparse-checkout (keep Windows CI)",
          "timestamp": "2026-07-01T23:58:32+08:00",
          "tree_id": "88d6f4882c96dee05748ec58a60fdebb350f1398",
          "url": "https://github.com/MolCrafts/molpy/commit/00978b0736346192ecf68c71f7ce8b6b63a01c14"
        },
        "date": 1782921569334,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 22623.098587997072,
            "unit": "iter/sec",
            "range": "stddev: 0.00000403749244349656",
            "extra": "mean: 44.20260982863598 usec\nrounds: 1343"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[small-1k]",
            "value": 10330.295010668922,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035080040023027398",
            "extra": "mean: 96.80265655213331 usec\nrounds: 7448"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[medium-10k]",
            "value": 1042.9332050462392,
            "unit": "iter/sec",
            "range": "stddev: 0.000011062245507758095",
            "extra": "mean: 958.8341757281227 usec\nrounds: 1030"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[large-100k]",
            "value": 169.152708866611,
            "unit": "iter/sec",
            "range": "stddev: 0.00003752931497169641",
            "extra": "mean: 5.911817828401268 msec\nrounds: 169"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[small-1k]",
            "value": 17039.616117363894,
            "unit": "iter/sec",
            "range": "stddev: 0.0000027982888186007753",
            "extra": "mean: 58.68676812389977 usec\nrounds: 10704"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[medium-10k]",
            "value": 1735.9681897418666,
            "unit": "iter/sec",
            "range": "stddev: 0.000009592955838223608",
            "extra": "mean: 576.0474217840921 usec\nrounds: 1726"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[large-100k]",
            "value": 106.7569721871684,
            "unit": "iter/sec",
            "range": "stddev: 0.000056466584764149156",
            "extra": "mean: 9.367069705262722 msec\nrounds: 95"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[small-1k]",
            "value": 4014.663534574879,
            "unit": "iter/sec",
            "range": "stddev: 0.000007652879924034736",
            "extra": "mean: 249.08687649359686 usec\nrounds: 2761"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[medium-10k]",
            "value": 464.6670942388942,
            "unit": "iter/sec",
            "range": "stddev: 0.00007100105282307275",
            "extra": "mean: 2.152078364055366 msec\nrounds: 434"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[large-100k]",
            "value": 49.28237530798252,
            "unit": "iter/sec",
            "range": "stddev: 0.0031898434421741774",
            "extra": "mean: 20.291229749999992 msec\nrounds: 44"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[small-1k]",
            "value": 75013.29665533188,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015641270699657677",
            "extra": "mean: 13.33096990250622 usec\nrounds: 7376"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[medium-10k]",
            "value": 69037.80812061879,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013256862498342657",
            "extra": "mean: 14.484816757983667 usec\nrounds: 21327"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[large-100k]",
            "value": 29030.813779070028,
            "unit": "iter/sec",
            "range": "stddev: 0.000002750491820435582",
            "extra": "mean: 34.44615805847499 usec\nrounds: 11249"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[small-1k]",
            "value": 319035.58625300834,
            "unit": "iter/sec",
            "range": "stddev: 6.961380002754784e-7",
            "extra": "mean: 3.1344465730132027 usec\nrounds: 30640"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[medium-10k]",
            "value": 319177.3804620394,
            "unit": "iter/sec",
            "range": "stddev: 5.821415436905976e-7",
            "extra": "mean: 3.1330540984840645 usec\nrounds: 95548"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[large-100k]",
            "value": 321064.9161687735,
            "unit": "iter/sec",
            "range": "stddev: 5.910692152398879e-7",
            "extra": "mean: 3.114634921600503 usec\nrounds: 77167"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[small-1k]",
            "value": 42.76623869578963,
            "unit": "iter/sec",
            "range": "stddev: 0.00569001861947497",
            "extra": "mean: 23.382930800001606 msec\nrounds: 45"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[medium-10k]",
            "value": 4.154129479664805,
            "unit": "iter/sec",
            "range": "stddev: 0.0013382270165379607",
            "extra": "mean: 240.7243213999891 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[large-50k]",
            "value": 0.7207021025571042,
            "unit": "iter/sec",
            "range": "stddev: 0.06788216912617574",
            "extra": "mean: 1.387535843799992 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[small-1k]",
            "value": 364.2725407145938,
            "unit": "iter/sec",
            "range": "stddev: 0.0028885742894276143",
            "extra": "mean: 2.7451973130840415 msec\nrounds: 428"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[medium-10k]",
            "value": 32.51307399474351,
            "unit": "iter/sec",
            "range": "stddev: 0.008728615496843065",
            "extra": "mean: 30.75685800000557 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[large-50k]",
            "value": 5.664884879904703,
            "unit": "iter/sec",
            "range": "stddev: 0.01482107314216127",
            "extra": "mean: 176.52609385714868 msec\nrounds: 7"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "42854324+Roy-Kid@users.noreply.github.com",
            "name": "Jichen Li",
            "username": "Roy-Kid"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b90fa48527d0e3f402f29bc2a461a23f5092944b",
          "message": "Merge pull request #40: Release molpy 0.6.0\n\nRelease molpy 0.6.0",
          "timestamp": "2026-07-04T15:02:35+08:00",
          "tree_id": "c5b2d0c9fea568e33539d24099791a206aa6931a",
          "url": "https://github.com/MolCrafts/molpy/commit/b90fa48527d0e3f402f29bc2a461a23f5092944b"
        },
        "date": 1783148621275,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster",
            "value": 13168.094891055312,
            "unit": "iter/sec",
            "range": "stddev: 0.0000023366789870459076",
            "extra": "mean: 75.94112954633017 usec\nrounds: 5334"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_centers",
            "value": 114045.55539626908,
            "unit": "iter/sec",
            "range": "stddev: 7.855648495970154e-7",
            "extra": "mean: 8.76842588494873 usec\nrounds: 31640"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_properties",
            "value": 48839.21072598396,
            "unit": "iter/sec",
            "range": "stddev: 9.263274025351486e-7",
            "extra": "mean: 20.475351364922226 usec\nrounds: 19891"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_van_hove",
            "value": 67.31729452197094,
            "unit": "iter/sec",
            "range": "stddev: 0.0012574358133897333",
            "extra": "mean: 14.855023617647339 msec\nrounds: 68"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_legendre_reorientation",
            "value": 42381.738914376765,
            "unit": "iter/sec",
            "range": "stddev: 0.000003967271772199533",
            "extra": "mean: 23.59506772528343 usec\nrounds: 14736"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_local_density",
            "value": 26793.652396872312,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022872625963799146",
            "extra": "mean: 37.322272648305784 usec\nrounds: 10365"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_gaussian_density",
            "value": 346.6973079824697,
            "unit": "iter/sec",
            "range": "stddev: 0.00018466133550514352",
            "extra": "mean: 2.884360440579376 msec\nrounds: 345"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_acf_analyzer",
            "value": 4095.518675767483,
            "unit": "iter/sec",
            "range": "stddev: 0.000022532203679367836",
            "extra": "mean: 244.1693175315833 usec\nrounds: 2447"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_spectral_analyzer",
            "value": 509667.1401837861,
            "unit": "iter/sec",
            "range": "stddev: 2.696459623265208e-7",
            "extra": "mean: 1.9620648873682531 usec\nrounds: 77072"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_dielectric_susceptibility",
            "value": 695.6347544286557,
            "unit": "iter/sec",
            "range": "stddev: 0.00003312741861152656",
            "extra": "mean: 1.4375359966327848 msec\nrounds: 594"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_debye_fit",
            "value": 56939.60729596819,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011956042616184934",
            "extra": "mean: 17.562467454369123 usec\nrounds: 14303"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_ionic_conductivity",
            "value": 369.80956734355897,
            "unit": "iter/sec",
            "range": "stddev: 0.000041144009340049924",
            "extra": "mean: 2.7040944537570173 msec\nrounds: 346"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_distance_distribution",
            "value": 216979.07849709113,
            "unit": "iter/sec",
            "range": "stddev: 5.363724926312794e-7",
            "extra": "mean: 4.608739270746816 usec\nrounds: 37188"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_angle_distribution",
            "value": 126981.92340458691,
            "unit": "iter/sec",
            "range": "stddev: 5.751458027530185e-7",
            "extra": "mean: 7.875136658733879 usec\nrounds: 34041"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_dihedral_distribution",
            "value": 150651.05230090598,
            "unit": "iter/sec",
            "range": "stddev: 6.057317446442426e-7",
            "extra": "mean: 6.637856056940309 usec\nrounds: 39071"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_combined_distribution",
            "value": 78150.05838299789,
            "unit": "iter/sec",
            "range": "stddev: 8.113069213663634e-7",
            "extra": "mean: 12.795895750956436 usec\nrounds: 19722"
          },
          {
            "name": "benchmarks/compute/test_hbond.py::test_hbonds",
            "value": 4018.548063947517,
            "unit": "iter/sec",
            "range": "stddev: 0.000007184452382473168",
            "extra": "mean: 248.8460966714619 usec\nrounds: 2824"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_pca",
            "value": 13754.25754575337,
            "unit": "iter/sec",
            "range": "stddev: 0.0000028801159083021518",
            "extra": "mean: 72.70476044770226 usec\nrounds: 9739"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_kmeans",
            "value": 13113.84531707039,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019970521196799455",
            "extra": "mean: 76.25528407737832 usec\nrounds: 10036"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_steinhardt",
            "value": 83.71674368309242,
            "unit": "iter/sec",
            "range": "stddev: 0.000053795709064747024",
            "extra": "mean: 11.945041768292784 msec\nrounds: 82"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_hexatic",
            "value": 1383.3861584413148,
            "unit": "iter/sec",
            "range": "stddev: 0.00000663979391601152",
            "extra": "mean: 722.8639623853959 usec\nrounds: 1090"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_solid_liquid",
            "value": 145.56312775507791,
            "unit": "iter/sec",
            "range": "stddev: 0.000037328793970782775",
            "extra": "mean: 6.869871618055523 msec\nrounds: 144"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_nematic",
            "value": 50706.79797322382,
            "unit": "iter/sec",
            "range": "stddev: 9.852382286662984e-7",
            "extra": "mean: 19.721221610720896 usec\nrounds: 25441"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_neighborlist",
            "value": 1003.4017754122075,
            "unit": "iter/sec",
            "range": "stddev: 0.00009927453198994107",
            "extra": "mean: 996.6097574316032 usec\nrounds: 841"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_rdf",
            "value": 1621.3565236065065,
            "unit": "iter/sec",
            "range": "stddev: 0.000009800673041399601",
            "extra": "mean: 616.7674940337145 usec\nrounds: 1257"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_center_of_mass",
            "value": 113462.32300347443,
            "unit": "iter/sec",
            "range": "stddev: 7.613569920344988e-7",
            "extra": "mean: 8.813498380157244 usec\nrounds: 50006"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_gyration_tensor",
            "value": 95878.01523258195,
            "unit": "iter/sec",
            "range": "stddev: 6.410956496709551e-7",
            "extra": "mean: 10.429919701343305 usec\nrounds: 47672"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_inertia_tensor",
            "value": 82515.78595509518,
            "unit": "iter/sec",
            "range": "stddev: 6.455769719327548e-7",
            "extra": "mean: 12.11889323267425 usec\nrounds: 44892"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_radius_of_gyration",
            "value": 109374.18642373853,
            "unit": "iter/sec",
            "range": "stddev: 6.151272666375467e-7",
            "extra": "mean: 9.14292515169704 usec\nrounds: 57169"
          },
          {
            "name": "benchmarks/compute/test_spatial.py::test_spatial_distribution",
            "value": 44374.57171131643,
            "unit": "iter/sec",
            "range": "stddev: 9.522945488733019e-7",
            "extra": "mean: 22.535428770008377 usec\nrounds: 16552"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_power_spectrum",
            "value": 176214.6057577516,
            "unit": "iter/sec",
            "range": "stddev: 5.860860617031865e-7",
            "extra": "mean: 5.674898489258802 usec\nrounds: 29721"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_ir_spectrum",
            "value": 174490.0409618923,
            "unit": "iter/sec",
            "range": "stddev: 4.368823237698898e-7",
            "extra": "mean: 5.730986103776518 usec\nrounds: 37996"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_vcd_spectrum",
            "value": 176453.78639971622,
            "unit": "iter/sec",
            "range": "stddev: 4.704925405772355e-7",
            "extra": "mean: 5.66720624364912 usec\nrounds: 37606"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_raman_spectrum",
            "value": 158358.70956749443,
            "unit": "iter/sec",
            "range": "stddev: 5.062342890293766e-7",
            "extra": "mean: 6.314777398295152 usec\nrounds: 36891"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_roa_spectrum",
            "value": 160822.25680064436,
            "unit": "iter/sec",
            "range": "stddev: 4.965981758498485e-7",
            "extra": "mean: 6.218044814777112 usec\nrounds: 36305"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_resonance_raman_spectrum",
            "value": 170337.95330771658,
            "unit": "iter/sec",
            "range": "stddev: 4.5455690786067215e-7",
            "extra": "mean: 5.870682255959092 usec\nrounds: 38937"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_static_structure_factor",
            "value": 23.3910565601384,
            "unit": "iter/sec",
            "range": "stddev: 0.003906625114844373",
            "extra": "mean: 42.75138224000273 msec\nrounds: 25"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_bond_order",
            "value": 937.1171691404603,
            "unit": "iter/sec",
            "range": "stddev: 0.000011563851993814128",
            "extra": "mean: 1.0671024210528732 msec\nrounds: 760"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_pmft_xy",
            "value": 6307.883230146071,
            "unit": "iter/sec",
            "range": "stddev: 0.000003321076008583584",
            "extra": "mean: 158.53178689499032 usec\nrounds: 5082"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_msd",
            "value": 26492.681476726968,
            "unit": "iter/sec",
            "range": "stddev: 0.000006391320306811253",
            "extra": "mean: 37.746273470976135 usec\nrounds: 7946"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_mcd",
            "value": 1762.398912154542,
            "unit": "iter/sec",
            "range": "stddev: 0.00001505373967756416",
            "extra": "mean: 567.4084301252177 usec\nrounds: 1195"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_pmsd",
            "value": 1576.0456235611525,
            "unit": "iter/sec",
            "range": "stddev: 0.000028234672550973512",
            "extra": "mean: 634.4993983996801 usec\nrounds: 1250"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_onsager",
            "value": 3093.5965695270743,
            "unit": "iter/sec",
            "range": "stddev: 0.000009594605810296296",
            "extra": "mean: 323.248354310424 usec\nrounds: 2320"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_jacf",
            "value": 1932.376101863322,
            "unit": "iter/sec",
            "range": "stddev: 0.000011280581614149692",
            "extra": "mean: 517.4976025814723 usec\nrounds: 1472"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_persist",
            "value": 6799.922943598626,
            "unit": "iter/sec",
            "range": "stddev: 0.000005123948254373689",
            "extra": "mean: 147.0604899929622 usec\nrounds: 4247"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_radical_voronoi",
            "value": 104.37429733074335,
            "unit": "iter/sec",
            "range": "stddev: 0.0001917957118486497",
            "extra": "mean: 9.580902823529247 msec\nrounds: 85"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_domains",
            "value": 5345.545472694745,
            "unit": "iter/sec",
            "range": "stddev: 0.000006511232420397511",
            "extra": "mean: 187.0716478062041 usec\nrounds: 3305"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_voids",
            "value": 13064.901394325745,
            "unit": "iter/sec",
            "range": "stddev: 0.000002651004540155687",
            "extra": "mean: 76.54095272654051 usec\nrounds: 10471"
          },
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 20587.434618092204,
            "unit": "iter/sec",
            "range": "stddev: 0.000005673122545061686",
            "extra": "mean: 48.5733175866993 usec\nrounds: 3895"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[reg-1k]",
            "value": 16726.023027448937,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024615810478135114",
            "extra": "mean: 59.7870754069218 usec\nrounds: 11365"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[reg-1k]",
            "value": 16612.487415310166,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024127607949478814",
            "extra": "mean: 60.195681417244835 usec\nrounds: 14169"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[reg-1k]",
            "value": 5746.854503291672,
            "unit": "iter/sec",
            "range": "stddev: 0.000004600633327086894",
            "extra": "mean: 174.00823344791868 usec\nrounds: 4078"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[reg-1k]",
            "value": 80021.92351092472,
            "unit": "iter/sec",
            "range": "stddev: 8.696337945118786e-7",
            "extra": "mean: 12.496575389911472 usec\nrounds: 15002"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[reg-1k]",
            "value": 346600.0200191098,
            "unit": "iter/sec",
            "range": "stddev: 3.617596965337997e-7",
            "extra": "mean: 2.8851700584000683 usec\nrounds: 84912"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[reg-1k]",
            "value": 43.945599086019776,
            "unit": "iter/sec",
            "range": "stddev: 0.007066658445738192",
            "extra": "mean: 22.755407157895036 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[reg-1k]",
            "value": 380.90637461030855,
            "unit": "iter/sec",
            "range": "stddev: 0.003319040941815213",
            "extra": "mean: 2.6253170507399974 msec\nrounds: 473"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "42854324+Roy-Kid@users.noreply.github.com",
            "name": "Jichen Li",
            "username": "Roy-Kid"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c17ccc6b6e3c6a1872006d53f95bc0a6a67edb21",
          "message": "Merge pull request #41 from MolCrafts/nightly\n\nRelease molpy 0.7.0",
          "timestamp": "2026-07-08T10:57:29+08:00",
          "tree_id": "07ffe820f002f9819fd9fedf39f659b287dc91fa",
          "url": "https://github.com/MolCrafts/molpy/commit/c17ccc6b6e3c6a1872006d53f95bc0a6a67edb21"
        },
        "date": 1783481188933,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster",
            "value": 7826.655456611844,
            "unit": "iter/sec",
            "range": "stddev: 0.000004932441962875351",
            "extra": "mean: 127.76849645977639 usec\nrounds: 4802"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_centers",
            "value": 93297.4734234125,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015431187243908907",
            "extra": "mean: 10.718403867827094 usec\nrounds: 51502"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_properties",
            "value": 48149.747721397805,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017095439017463339",
            "extra": "mean: 20.768540798721546 usec\nrounds: 19155"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_van_hove",
            "value": 63.165236266028224,
            "unit": "iter/sec",
            "range": "stddev: 0.0009359556944399249",
            "extra": "mean: 15.831493066666862 msec\nrounds: 60"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_legendre_reorientation",
            "value": 33566.65024885197,
            "unit": "iter/sec",
            "range": "stddev: 0.000007327278303170327",
            "extra": "mean: 29.791474352857165 usec\nrounds: 10469"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_local_density",
            "value": 18786.055604125846,
            "unit": "iter/sec",
            "range": "stddev: 0.000004630609012411417",
            "extra": "mean: 53.23097200778951 usec\nrounds: 11789"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_gaussian_density",
            "value": 365.4255336875144,
            "unit": "iter/sec",
            "range": "stddev: 0.00002774153839844806",
            "extra": "mean: 2.736535649025358 msec\nrounds: 359"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_acf_analyzer",
            "value": 3628.277025263121,
            "unit": "iter/sec",
            "range": "stddev: 0.000014640826280952004",
            "extra": "mean: 275.61291297140696 usec\nrounds: 2413"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_spectral_analyzer",
            "value": 468084.9507792974,
            "unit": "iter/sec",
            "range": "stddev: 5.264288997262367e-7",
            "extra": "mean: 2.13636434654679 usec\nrounds: 69508"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_dielectric_susceptibility",
            "value": 555.5618424460084,
            "unit": "iter/sec",
            "range": "stddev: 0.00003770850980292169",
            "extra": "mean: 1.7999796307054399 msec\nrounds: 482"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_debye_fit",
            "value": 52591.3183629636,
            "unit": "iter/sec",
            "range": "stddev: 0.0000030723914099693073",
            "extra": "mean: 19.014545197334897 usec\nrounds: 12722"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_ionic_conductivity",
            "value": 289.35892436930806,
            "unit": "iter/sec",
            "range": "stddev: 0.00041979319785647763",
            "extra": "mean: 3.4559155283688523 msec\nrounds: 282"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_distance_distribution",
            "value": 217026.37048664235,
            "unit": "iter/sec",
            "range": "stddev: 8.427483732584331e-7",
            "extra": "mean: 4.6077349851895 usec\nrounds: 54363"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_angle_distribution",
            "value": 119964.06114934671,
            "unit": "iter/sec",
            "range": "stddev: 0.000001729108664467642",
            "extra": "mean: 8.33582983452912 usec\nrounds: 36441"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_dihedral_distribution",
            "value": 144341.27412757126,
            "unit": "iter/sec",
            "range": "stddev: 9.826799582281276e-7",
            "extra": "mean: 6.9280253069969655 usec\nrounds: 41451"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_combined_distribution",
            "value": 75142.33780722767,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013886242831117165",
            "extra": "mean: 13.30807676712733 usec\nrounds: 29101"
          },
          {
            "name": "benchmarks/compute/test_hbond.py::test_hbonds",
            "value": 3800.677473043657,
            "unit": "iter/sec",
            "range": "stddev: 0.000008508821823275596",
            "extra": "mean: 263.11098668395573 usec\nrounds: 3079"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_pca",
            "value": 15178.591422847074,
            "unit": "iter/sec",
            "range": "stddev: 0.000003639080504177093",
            "extra": "mean: 65.88226615644868 usec\nrounds: 11760"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_kmeans",
            "value": 17542.176374756906,
            "unit": "iter/sec",
            "range": "stddev: 0.000009303523465514394",
            "extra": "mean: 57.005469483193345 usec\nrounds: 12616"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_steinhardt",
            "value": 79.09200212183234,
            "unit": "iter/sec",
            "range": "stddev: 0.00004337737044459171",
            "extra": "mean: 12.643503428571861 msec\nrounds: 77"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_hexatic",
            "value": 1269.7757242755647,
            "unit": "iter/sec",
            "range": "stddev: 0.00000703895215657085",
            "extra": "mean: 787.5406505904988 usec\nrounds: 1016"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_solid_liquid",
            "value": 137.50842534850125,
            "unit": "iter/sec",
            "range": "stddev: 0.000026978533706112018",
            "extra": "mean: 7.272281661764366 msec\nrounds: 136"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_nematic",
            "value": 44123.656926734846,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017627399776023422",
            "extra": "mean: 22.66357935065198 usec\nrounds: 26364"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_neighborlist",
            "value": 1054.7186515430797,
            "unit": "iter/sec",
            "range": "stddev: 0.00009984027029243929",
            "extra": "mean: 948.1201442081024 usec\nrounds: 846"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_rdf",
            "value": 1260.89812069185,
            "unit": "iter/sec",
            "range": "stddev: 0.000012416586316025378",
            "extra": "mean: 793.0854869157105 usec\nrounds: 1070"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_center_of_mass",
            "value": 108573.36523846388,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010592231228972243",
            "extra": "mean: 9.210362023905784 usec\nrounds: 53955"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_gyration_tensor",
            "value": 93111.0826030859,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011895182528229346",
            "extra": "mean: 10.739860090154917 usec\nrounds: 48810"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_inertia_tensor",
            "value": 79831.69566112,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013217102225695486",
            "extra": "mean: 12.526352994491441 usec\nrounds: 49658"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_radius_of_gyration",
            "value": 101836.71913978973,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010777156756384749",
            "extra": "mean: 9.81964077836517 usec\nrounds: 57761"
          },
          {
            "name": "benchmarks/compute/test_spatial.py::test_spatial_distribution",
            "value": 40246.643441849075,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019913569183200437",
            "extra": "mean: 24.846792539230357 usec\nrounds: 16379"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_power_spectrum",
            "value": 177153.42335372357,
            "unit": "iter/sec",
            "range": "stddev: 9.13886157526948e-7",
            "extra": "mean: 5.644824588025559 usec\nrounds: 31617"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_ir_spectrum",
            "value": 177179.96185777764,
            "unit": "iter/sec",
            "range": "stddev: 8.849933268135969e-7",
            "extra": "mean: 5.643979090607887 usec\nrounds: 47921"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_vcd_spectrum",
            "value": 175406.20937330104,
            "unit": "iter/sec",
            "range": "stddev: 9.043243831568102e-7",
            "extra": "mean: 5.70105245175096 usec\nrounds: 38054"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_raman_spectrum",
            "value": 158441.86508619023,
            "unit": "iter/sec",
            "range": "stddev: 9.66594565773378e-7",
            "extra": "mean: 6.311463194756093 usec\nrounds: 42820"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_roa_spectrum",
            "value": 169120.01945788827,
            "unit": "iter/sec",
            "range": "stddev: 9.227371001672191e-7",
            "extra": "mean: 5.912960530666241 usec\nrounds: 45453"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_resonance_raman_spectrum",
            "value": 180328.98759851686,
            "unit": "iter/sec",
            "range": "stddev: 8.820574485094563e-7",
            "extra": "mean: 5.545420141915246 usec\nrounds: 46927"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_static_structure_factor",
            "value": 22.7253332368514,
            "unit": "iter/sec",
            "range": "stddev: 0.0002983560891486726",
            "extra": "mean: 44.00375517391314 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_bond_order",
            "value": 858.2031694543494,
            "unit": "iter/sec",
            "range": "stddev: 0.000013097609816807665",
            "extra": "mean: 1.165225246879251 msec\nrounds: 721"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_pmft_xy",
            "value": 4509.038398898271,
            "unit": "iter/sec",
            "range": "stddev: 0.000006818343430022483",
            "extra": "mean: 221.77677622890457 usec\nrounds: 4232"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_msd",
            "value": 34770.30947794308,
            "unit": "iter/sec",
            "range": "stddev: 0.0000047882466575349186",
            "extra": "mean: 28.76016966815785 usec\nrounds: 8829"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_mcd",
            "value": 1455.494498028088,
            "unit": "iter/sec",
            "range": "stddev: 0.000024125990750188945",
            "extra": "mean: 687.0517211537424 usec\nrounds: 1144"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_pmsd",
            "value": 1297.7732088036473,
            "unit": "iter/sec",
            "range": "stddev: 0.000026462563990059287",
            "extra": "mean: 770.550658016627 usec\nrounds: 1079"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_onsager",
            "value": 2576.872416955739,
            "unit": "iter/sec",
            "range": "stddev: 0.000019699139608737743",
            "extra": "mean: 388.0673305438141 usec\nrounds: 2151"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_jacf",
            "value": 1482.7926941294104,
            "unit": "iter/sec",
            "range": "stddev: 0.000028802707175229894",
            "extra": "mean: 674.4031070284766 usec\nrounds: 1252"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_persist",
            "value": 5553.625727981303,
            "unit": "iter/sec",
            "range": "stddev: 0.000021837728260150157",
            "extra": "mean: 180.0625481406886 usec\nrounds: 3926"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_radical_voronoi",
            "value": 110.45937995297861,
            "unit": "iter/sec",
            "range": "stddev: 0.0001265083471276266",
            "extra": "mean: 9.053101696077684 msec\nrounds: 102"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_domains",
            "value": 5749.0962455919425,
            "unit": "iter/sec",
            "range": "stddev: 0.000016107077981171976",
            "extra": "mean: 173.940382502161 usec\nrounds: 3749"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_voids",
            "value": 10459.836157176556,
            "unit": "iter/sec",
            "range": "stddev: 0.000004856223835526731",
            "extra": "mean: 95.60379196894915 usec\nrounds: 7720"
          },
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 16615.353488151486,
            "unit": "iter/sec",
            "range": "stddev: 0.0000075616329449903075",
            "extra": "mean: 60.185297936219435 usec\nrounds: 4991"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[reg-1k]",
            "value": 14423.064850306197,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035231467896110042",
            "extra": "mean: 69.3333913685322 usec\nrounds: 12883"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[reg-1k]",
            "value": 15285.973089003397,
            "unit": "iter/sec",
            "range": "stddev: 0.0000032001418581596654",
            "extra": "mean: 65.41945312721974 usec\nrounds: 14102"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[reg-1k]",
            "value": 4793.348198523063,
            "unit": "iter/sec",
            "range": "stddev: 0.000010683251553685931",
            "extra": "mean: 208.62244063724023 usec\nrounds: 3327"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[reg-1k]",
            "value": 77652.40210394672,
            "unit": "iter/sec",
            "range": "stddev: 0.000002350673897150595",
            "extra": "mean: 12.877901686304364 usec\nrounds: 13996"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[reg-1k]",
            "value": 337542.4914478969,
            "unit": "iter/sec",
            "range": "stddev: 6.726065804733492e-7",
            "extra": "mean: 2.9625899711484474 usec\nrounds: 80109"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[reg-1k]",
            "value": 40.756115214574656,
            "unit": "iter/sec",
            "range": "stddev: 0.006549958223831778",
            "extra": "mean: 24.53619523684125 msec\nrounds: 38"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[reg-1k]",
            "value": 348.94811694718624,
            "unit": "iter/sec",
            "range": "stddev: 0.003086869500261501",
            "extra": "mean: 2.8657555419660037 msec\nrounds: 417"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "42854324+Roy-Kid@users.noreply.github.com",
            "name": "Jichen Li",
            "username": "Roy-Kid"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "80180100a8561fe2cade60f6aacac24cf0e5b5af",
          "message": "release: v0.8.0 with molrs core cutover\n\nrelease: v0.8.0 with molrs core cutover",
          "timestamp": "2026-07-17T15:04:15+08:00",
          "tree_id": "2c73f71a27cd40a0ebf0ae1657dcb7733816f07d",
          "url": "https://github.com/MolCrafts/molpy/commit/80180100a8561fe2cade60f6aacac24cf0e5b5af"
        },
        "date": 1784271923368,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster",
            "value": 7798.7356232275915,
            "unit": "iter/sec",
            "range": "stddev: 0.000005070987827201252",
            "extra": "mean: 128.22591357265924 usec\nrounds: 4929"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_centers",
            "value": 113037.50901509088,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011946229035938556",
            "extra": "mean: 8.846620990794275 usec\nrounds: 24163"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_properties",
            "value": 49112.863544770626,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017348993709298827",
            "extra": "mean: 20.361264398448554 usec\nrounds: 18752"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_van_hove",
            "value": 60.933761992225214,
            "unit": "iter/sec",
            "range": "stddev: 0.0008781950050310651",
            "extra": "mean: 16.411263104477186 msec\nrounds: 67"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_legendre_reorientation",
            "value": 33118.846620446384,
            "unit": "iter/sec",
            "range": "stddev: 0.000007582559081339594",
            "extra": "mean: 30.194288208775845 usec\nrounds: 10194"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_local_density",
            "value": 18504.95495761292,
            "unit": "iter/sec",
            "range": "stddev: 0.000004042478026697646",
            "extra": "mean: 54.03958033351499 usec\nrounds: 12112"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_gaussian_density",
            "value": 365.2448122588015,
            "unit": "iter/sec",
            "range": "stddev: 0.00002113276327773237",
            "extra": "mean: 2.737889674094618 msec\nrounds: 359"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_acf_analyzer",
            "value": 3778.906662093879,
            "unit": "iter/sec",
            "range": "stddev: 0.000014656094305158643",
            "extra": "mean: 264.62680595712396 usec\nrounds: 2283"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_spectral_analyzer",
            "value": 481341.92526582704,
            "unit": "iter/sec",
            "range": "stddev: 5.270435664983538e-7",
            "extra": "mean: 2.07752524247236 usec\nrounds: 71348"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_dielectric_susceptibility",
            "value": 835.7438147873004,
            "unit": "iter/sec",
            "range": "stddev: 0.000029220374485205695",
            "extra": "mean: 1.1965389181546062 msec\nrounds: 672"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_debye_fit",
            "value": 52466.87144948761,
            "unit": "iter/sec",
            "range": "stddev: 0.0000026206354426091048",
            "extra": "mean: 19.059646065665422 usec\nrounds: 12734"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_ionic_conductivity",
            "value": 453.1001997240627,
            "unit": "iter/sec",
            "range": "stddev: 0.00004024826547449281",
            "extra": "mean: 2.2070173454105704 msec\nrounds: 414"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_distance_distribution",
            "value": 217271.26301480265,
            "unit": "iter/sec",
            "range": "stddev: 7.85933083203881e-7",
            "extra": "mean: 4.602541477985841 usec\nrounds: 49906"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_angle_distribution",
            "value": 122848.78520961516,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011830129131300955",
            "extra": "mean: 8.140088632490048 usec\nrounds: 31027"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_dihedral_distribution",
            "value": 151396.0262375101,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010597290442391664",
            "extra": "mean: 6.605193180111609 usec\nrounds: 46599"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_combined_distribution",
            "value": 76066.5619522305,
            "unit": "iter/sec",
            "range": "stddev: 0.000001301573178920835",
            "extra": "mean: 13.14638093710606 usec\nrounds: 31202"
          },
          {
            "name": "benchmarks/compute/test_hbond.py::test_hbonds",
            "value": 3565.4946848792274,
            "unit": "iter/sec",
            "range": "stddev: 0.0000217026327642238",
            "extra": "mean: 280.46599094393895 usec\nrounds: 2871"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_pca",
            "value": 15272.790713112094,
            "unit": "iter/sec",
            "range": "stddev: 0.000003824595973712281",
            "extra": "mean: 65.47591850004686 usec\nrounds: 12000"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_kmeans",
            "value": 16966.85020004983,
            "unit": "iter/sec",
            "range": "stddev: 0.000004034712845544282",
            "extra": "mean: 58.93845871268805 usec\nrounds: 12413"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_steinhardt",
            "value": 73.98116116261315,
            "unit": "iter/sec",
            "range": "stddev: 0.0018223132205452137",
            "extra": "mean: 13.516954644736725 msec\nrounds: 76"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_hexatic",
            "value": 1269.3973341628496,
            "unit": "iter/sec",
            "range": "stddev: 0.000008692141599699974",
            "extra": "mean: 787.7754057672466 usec\nrounds: 971"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_solid_liquid",
            "value": 137.8139860546702,
            "unit": "iter/sec",
            "range": "stddev: 0.00022767983596380323",
            "extra": "mean: 7.256157583333408 msec\nrounds: 132"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_nematic",
            "value": 44500.931337291964,
            "unit": "iter/sec",
            "range": "stddev: 0.0000018015727031199545",
            "extra": "mean: 22.471439809215767 usec\nrounds: 24954"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_neighborlist",
            "value": 1090.5099679225864,
            "unit": "iter/sec",
            "range": "stddev: 0.00008781326145809483",
            "extra": "mean: 917.0021635886491 usec\nrounds: 758"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_rdf",
            "value": 1247.0308323963427,
            "unit": "iter/sec",
            "range": "stddev: 0.000012654833474283649",
            "extra": "mean: 801.9047917831841 usec\nrounds: 1071"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_center_of_mass",
            "value": 105977.29252295688,
            "unit": "iter/sec",
            "range": "stddev: 0.0000012890272065892125",
            "extra": "mean: 9.435983654549199 usec\nrounds: 54633"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_gyration_tensor",
            "value": 91731.31327234517,
            "unit": "iter/sec",
            "range": "stddev: 0.0000015358090468025965",
            "extra": "mean: 10.901402850638968 usec\nrounds: 40623"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_inertia_tensor",
            "value": 78319.41455680791,
            "unit": "iter/sec",
            "range": "stddev: 0.000001272817019432317",
            "extra": "mean: 12.768226188343936 usec\nrounds: 50259"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_radius_of_gyration",
            "value": 102613.71619037306,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011184629989974478",
            "extra": "mean: 9.745285885024963 usec\nrounds: 58576"
          },
          {
            "name": "benchmarks/compute/test_spatial.py::test_spatial_distribution",
            "value": 40205.79806194747,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022782048594785728",
            "extra": "mean: 24.872034587131946 usec\nrounds: 16191"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_power_spectrum",
            "value": 177996.9183961199,
            "unit": "iter/sec",
            "range": "stddev: 9.484761663954401e-7",
            "extra": "mean: 5.618074790343105 usec\nrounds: 16339"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_ir_spectrum",
            "value": 179889.6237889417,
            "unit": "iter/sec",
            "range": "stddev: 9.087099295208699e-7",
            "extra": "mean: 5.5589643189941045 usec\nrounds: 48289"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_vcd_spectrum",
            "value": 178150.71391634218,
            "unit": "iter/sec",
            "range": "stddev: 9.99876296814382e-7",
            "extra": "mean: 5.613224769167021 usec\nrounds: 45807"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_raman_spectrum",
            "value": 161910.33957373368,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010538223304843846",
            "extra": "mean: 6.176257814249112 usec\nrounds: 40087"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_roa_spectrum",
            "value": 170423.61297688424,
            "unit": "iter/sec",
            "range": "stddev: 9.421281458455298e-7",
            "extra": "mean: 5.867731487042451 usec\nrounds: 41417"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_resonance_raman_spectrum",
            "value": 181019.4401579704,
            "unit": "iter/sec",
            "range": "stddev: 8.883919521596513e-7",
            "extra": "mean: 5.52426854887701 usec\nrounds: 35420"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_static_structure_factor",
            "value": 23.40335087715104,
            "unit": "iter/sec",
            "range": "stddev: 0.00020935997989974617",
            "extra": "mean: 42.728923958334164 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_bond_order",
            "value": 850.6477836806818,
            "unit": "iter/sec",
            "range": "stddev: 0.000022150933629213978",
            "extra": "mean: 1.1755746845928214 msec\nrounds: 688"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_pmft_xy",
            "value": 5367.536202754314,
            "unit": "iter/sec",
            "range": "stddev: 0.000011097866597189878",
            "extra": "mean: 186.30521755714602 usec\nrounds: 4192"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_msd",
            "value": 35342.45510828473,
            "unit": "iter/sec",
            "range": "stddev: 0.0000038260718447790895",
            "extra": "mean: 28.294582165730386 usec\nrounds: 7637"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_mcd",
            "value": 1498.887934794353,
            "unit": "iter/sec",
            "range": "stddev: 0.00004021803441680902",
            "extra": "mean: 667.1612845674148 usec\nrounds: 1121"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_pmsd",
            "value": 1332.6437369284308,
            "unit": "iter/sec",
            "range": "stddev: 0.000021383142634632555",
            "extra": "mean: 750.3880987013596 usec\nrounds: 1155"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_onsager",
            "value": 2606.2499509429763,
            "unit": "iter/sec",
            "range": "stddev: 0.000042250888337268156",
            "extra": "mean: 383.69305278574166 usec\nrounds: 2046"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_jacf",
            "value": 1487.8450898301514,
            "unit": "iter/sec",
            "range": "stddev: 0.000028614216540850468",
            "extra": "mean: 672.1129819463647 usec\nrounds: 1274"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_persist",
            "value": 5768.731979354455,
            "unit": "iter/sec",
            "range": "stddev: 0.000013669963307388441",
            "extra": "mean: 173.34832049380532 usec\nrounds: 4050"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_radical_voronoi",
            "value": 112.04362079634186,
            "unit": "iter/sec",
            "range": "stddev: 0.00006988270942951223",
            "extra": "mean: 8.925095359223247 msec\nrounds: 103"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_domains",
            "value": 5925.960340823212,
            "unit": "iter/sec",
            "range": "stddev: 0.00001815288605837742",
            "extra": "mean: 168.74901998771793 usec\nrounds: 3252"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_voids",
            "value": 11355.882444677001,
            "unit": "iter/sec",
            "range": "stddev: 0.000004287405661362746",
            "extra": "mean: 88.06008734871536 usec\nrounds: 8094"
          },
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 29753.241940317712,
            "unit": "iter/sec",
            "range": "stddev: 0.000005139247737721964",
            "extra": "mean: 33.60978282655412 usec\nrounds: 5788"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[reg-1k]",
            "value": 13716.307175246739,
            "unit": "iter/sec",
            "range": "stddev: 0.000008571617106621318",
            "extra": "mean: 72.90592046557978 usec\nrounds: 12372"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[reg-1k]",
            "value": 14361.744445536558,
            "unit": "iter/sec",
            "range": "stddev: 0.0000036277093665202234",
            "extra": "mean: 69.62942446109231 usec\nrounds: 12199"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[reg-1k]",
            "value": 5343.731226347963,
            "unit": "iter/sec",
            "range": "stddev: 0.000008336105082167436",
            "extra": "mean: 187.13516036685562 usec\nrounds: 3598"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[reg-1k]",
            "value": 78599.41546576684,
            "unit": "iter/sec",
            "range": "stddev: 0.000002004171874548304",
            "extra": "mean: 12.722740927196076 usec\nrounds: 10526"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[reg-1k]",
            "value": 334027.651686433,
            "unit": "iter/sec",
            "range": "stddev: 6.497603795373891e-7",
            "extra": "mean: 2.9937641238718333 usec\nrounds: 79847"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[reg-1k]",
            "value": 1460.6589325154919,
            "unit": "iter/sec",
            "range": "stddev: 0.000015067003336427705",
            "extra": "mean: 684.6225205208157 usec\nrounds: 999"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[reg-1k]",
            "value": 195.79722864516452,
            "unit": "iter/sec",
            "range": "stddev: 0.002783829246428966",
            "extra": "mean: 5.107324587378405 msec\nrounds: 206"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "42854324+Roy-Kid@users.noreply.github.com",
            "name": "Jichen Li",
            "username": "Roy-Kid"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c06ea3190c58ce61b82c9da716d86f68907ba16c",
          "message": "refactor(io)!: drop HDF5/h5py; require zarr for MolStore\n\nrefactor(io)!: drop HDF5/h5py; require zarr for MolStore",
          "timestamp": "2026-07-17T15:24:19+08:00",
          "tree_id": "c9ebc1a4d7a58edc122d50f045871b28ede95054",
          "url": "https://github.com/MolCrafts/molpy/commit/c06ea3190c58ce61b82c9da716d86f68907ba16c"
        },
        "date": 1784273122471,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster",
            "value": 7815.759161437394,
            "unit": "iter/sec",
            "range": "stddev: 0.00000524100364720145",
            "extra": "mean: 127.94662416594862 usec\nrounds: 4196"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_centers",
            "value": 112583.50909627824,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011361406846125272",
            "extra": "mean: 8.882295533574355 usec\nrounds: 36764"
          },
          {
            "name": "benchmarks/compute/test_cluster.py::test_cluster_properties",
            "value": 48868.34384717401,
            "unit": "iter/sec",
            "range": "stddev: 0.0000022646310466120394",
            "extra": "mean: 20.46314487610426 usec\nrounds: 16559"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_van_hove",
            "value": 55.62475899047825,
            "unit": "iter/sec",
            "range": "stddev: 0.0010422037665655897",
            "extra": "mean: 17.97760598245789 msec\nrounds: 57"
          },
          {
            "name": "benchmarks/compute/test_correlation.py::test_legendre_reorientation",
            "value": 33556.74674413869,
            "unit": "iter/sec",
            "range": "stddev: 0.000007210936151911008",
            "extra": "mean: 29.800266623720564 usec\nrounds: 7715"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_local_density",
            "value": 18874.21893021785,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035774770690870565",
            "extra": "mean: 52.982324921482615 usec\nrounds: 10501"
          },
          {
            "name": "benchmarks/compute/test_density.py::test_gaussian_density",
            "value": 365.66556994559613,
            "unit": "iter/sec",
            "range": "stddev: 0.000017357425752487596",
            "extra": "mean: 2.7347392869084737 msec\nrounds: 359"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_acf_analyzer",
            "value": 3487.0571433077,
            "unit": "iter/sec",
            "range": "stddev: 0.000017269689510326626",
            "extra": "mean: 286.77476706086765 usec\nrounds: 2198"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_spectral_analyzer",
            "value": 457982.97747420776,
            "unit": "iter/sec",
            "range": "stddev: 5.316035044864468e-7",
            "extra": "mean: 2.1834872673980925 usec\nrounds: 66011"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_dielectric_susceptibility",
            "value": 801.1130494888903,
            "unit": "iter/sec",
            "range": "stddev: 0.00004680915615690912",
            "extra": "mean: 1.2482632764976171 msec\nrounds: 651"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_debye_fit",
            "value": 52159.464567609706,
            "unit": "iter/sec",
            "range": "stddev: 0.000003426608545040644",
            "extra": "mean: 19.171975945109413 usec\nrounds: 11349"
          },
          {
            "name": "benchmarks/compute/test_dielectric.py::test_ionic_conductivity",
            "value": 432.23573587945884,
            "unit": "iter/sec",
            "range": "stddev: 0.00011822405982995283",
            "extra": "mean: 2.3135523442209744 msec\nrounds: 398"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_distance_distribution",
            "value": 213888.38363352756,
            "unit": "iter/sec",
            "range": "stddev: 8.454073566901016e-7",
            "extra": "mean: 4.675335719556335 usec\nrounds: 47215"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_angle_distribution",
            "value": 123172.64931274451,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010949072178310074",
            "extra": "mean: 8.118685483990244 usec\nrounds: 25792"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_dihedral_distribution",
            "value": 149954.62290714576,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010097905770319303",
            "extra": "mean: 6.668684036631639 usec\nrounds: 41144"
          },
          {
            "name": "benchmarks/compute/test_distribution.py::test_combined_distribution",
            "value": 75086.95359381697,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013152727288382635",
            "extra": "mean: 13.31789281809863 usec\nrounds: 26469"
          },
          {
            "name": "benchmarks/compute/test_hbond.py::test_hbonds",
            "value": 3541.3577934939785,
            "unit": "iter/sec",
            "range": "stddev: 0.000008230126160725627",
            "extra": "mean: 282.3775676767692 usec\nrounds: 2475"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_pca",
            "value": 15213.595587195743,
            "unit": "iter/sec",
            "range": "stddev: 0.0000035014955349754554",
            "extra": "mean: 65.73068110484233 usec\nrounds: 10897"
          },
          {
            "name": "benchmarks/compute/test_ml.py::test_kmeans",
            "value": 17050.503334357873,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033076562614201793",
            "extra": "mean: 58.64929500262523 usec\nrounds: 9705"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_steinhardt",
            "value": 78.40433062187606,
            "unit": "iter/sec",
            "range": "stddev: 0.000060116298161650736",
            "extra": "mean: 12.754397519478141 msec\nrounds: 77"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_hexatic",
            "value": 1265.6843688129948,
            "unit": "iter/sec",
            "range": "stddev: 0.000009907077381481021",
            "extra": "mean: 790.0863948709714 usec\nrounds: 1170"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_solid_liquid",
            "value": 137.72817596536598,
            "unit": "iter/sec",
            "range": "stddev: 0.0000440205223725387",
            "extra": "mean: 7.260678455884484 msec\nrounds: 136"
          },
          {
            "name": "benchmarks/compute/test_order.py::test_nematic",
            "value": 44537.130829965856,
            "unit": "iter/sec",
            "range": "stddev: 0.000001990819063372307",
            "extra": "mean: 22.453175167879728 usec\nrounds: 25016"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_neighborlist",
            "value": 1085.2791896088886,
            "unit": "iter/sec",
            "range": "stddev: 0.00008145568825785487",
            "extra": "mean: 921.4218880953375 usec\nrounds: 840"
          },
          {
            "name": "benchmarks/compute/test_pair.py::test_rdf",
            "value": 1278.15531458506,
            "unit": "iter/sec",
            "range": "stddev: 0.000015179753598019186",
            "extra": "mean: 782.377531579282 usec\nrounds: 950"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_center_of_mass",
            "value": 104296.1459958864,
            "unit": "iter/sec",
            "range": "stddev: 0.0000017959130969758195",
            "extra": "mean: 9.58808199911281 usec\nrounds: 47781"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_gyration_tensor",
            "value": 93590.59635606442,
            "unit": "iter/sec",
            "range": "stddev: 0.0000014439392248088354",
            "extra": "mean: 10.684834149314646 usec\nrounds: 46361"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_inertia_tensor",
            "value": 79700.73219766752,
            "unit": "iter/sec",
            "range": "stddev: 0.0000013997238187867423",
            "extra": "mean: 12.546936175189437 usec\nrounds: 46424"
          },
          {
            "name": "benchmarks/compute/test_shape.py::test_radius_of_gyration",
            "value": 104164.71693818901,
            "unit": "iter/sec",
            "range": "stddev: 0.0000011140065642594465",
            "extra": "mean: 9.600179690339836 usec\nrounds: 55885"
          },
          {
            "name": "benchmarks/compute/test_spatial.py::test_spatial_distribution",
            "value": 40067.139047328375,
            "unit": "iter/sec",
            "range": "stddev: 0.0000024122541897185553",
            "extra": "mean: 24.95810840945677 usec\nrounds: 15340"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_power_spectrum",
            "value": 178212.644585257,
            "unit": "iter/sec",
            "range": "stddev: 0.000001026563377215455",
            "extra": "mean: 5.611274117654427 usec\nrounds: 16409"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_ir_spectrum",
            "value": 178597.69190772477,
            "unit": "iter/sec",
            "range": "stddev: 8.988527476994239e-7",
            "extra": "mean: 5.599176502889328 usec\nrounds: 27399"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_vcd_spectrum",
            "value": 177914.61452107003,
            "unit": "iter/sec",
            "range": "stddev: 8.62942806694961e-7",
            "extra": "mean: 5.620673729878285 usec\nrounds: 27183"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_raman_spectrum",
            "value": 160020.86241394727,
            "unit": "iter/sec",
            "range": "stddev: 9.748256733959596e-7",
            "extra": "mean: 6.2491851682011745 usec\nrounds: 32619"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_roa_spectrum",
            "value": 167531.81218526987,
            "unit": "iter/sec",
            "range": "stddev: 9.405443295869136e-7",
            "extra": "mean: 5.969015597432453 usec\nrounds: 35647"
          },
          {
            "name": "benchmarks/compute/test_spectra.py::test_resonance_raman_spectrum",
            "value": 177245.60859490946,
            "unit": "iter/sec",
            "range": "stddev: 0.0000010009888300375974",
            "extra": "mean: 5.6418887211218625 usec\nrounds: 28469"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_static_structure_factor",
            "value": 23.335999704660818,
            "unit": "iter/sec",
            "range": "stddev: 0.0003262165273341157",
            "extra": "mean: 42.85224599999774 msec\nrounds: 24"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_bond_order",
            "value": 849.6644146761003,
            "unit": "iter/sec",
            "range": "stddev: 0.000024656449205068693",
            "extra": "mean: 1.176935249643483 msec\nrounds: 701"
          },
          {
            "name": "benchmarks/compute/test_structure.py::test_pmft_xy",
            "value": 4561.306033828044,
            "unit": "iter/sec",
            "range": "stddev: 0.000007350406471547989",
            "extra": "mean: 219.23545418432647 usec\nrounds: 4027"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_msd",
            "value": 34523.48952409209,
            "unit": "iter/sec",
            "range": "stddev: 0.000004166861497231969",
            "extra": "mean: 28.965785724011294 usec\nrounds: 7257"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_mcd",
            "value": 1440.0896013572515,
            "unit": "iter/sec",
            "range": "stddev: 0.000027497263540346858",
            "extra": "mean: 694.4012366018911 usec\nrounds: 989"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_pmsd",
            "value": 1296.8180223593438,
            "unit": "iter/sec",
            "range": "stddev: 0.000027786143252437393",
            "extra": "mean: 771.1182160937792 usec\nrounds: 1106"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_onsager",
            "value": 2608.396288124682,
            "unit": "iter/sec",
            "range": "stddev: 0.00002069186458178744",
            "extra": "mean: 383.37732826592634 usec\nrounds: 2105"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_jacf",
            "value": 1488.6872792780716,
            "unit": "iter/sec",
            "range": "stddev: 0.000036990388869416285",
            "extra": "mean: 671.7327500003513 usec\nrounds: 1196"
          },
          {
            "name": "benchmarks/compute/test_transport.py::test_persist",
            "value": 5490.768952209131,
            "unit": "iter/sec",
            "range": "stddev: 0.00003092591631384329",
            "extra": "mean: 182.12385345365234 usec\nrounds: 3214"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_radical_voronoi",
            "value": 109.3536065334044,
            "unit": "iter/sec",
            "range": "stddev: 0.00042706565245991684",
            "extra": "mean: 9.144645811883017 msec\nrounds: 101"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_domains",
            "value": 5992.624030458729,
            "unit": "iter/sec",
            "range": "stddev: 0.000008073104306697469",
            "extra": "mean: 166.871806894158 usec\nrounds: 3133"
          },
          {
            "name": "benchmarks/compute/test_voronoi.py::test_voronoi_voids",
            "value": 10537.870781693748,
            "unit": "iter/sec",
            "range": "stddev: 0.000004911465725016883",
            "extra": "mean: 94.89583054455242 usec\nrounds: 7052"
          },
          {
            "name": "benchmarks/test_box.py::test_box_cubic_construct",
            "value": 29867.93988800803,
            "unit": "iter/sec",
            "range": "stddev: 0.0000050006846177302185",
            "extra": "mean: 33.48071556825048 usec\nrounds: 6571"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_fractional[reg-1k]",
            "value": 13939.860332662782,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033664648365814506",
            "extra": "mean: 71.73673022080995 usec\nrounds: 12273"
          },
          {
            "name": "benchmarks/test_box.py::test_box_make_absolute[reg-1k]",
            "value": 14192.273061123085,
            "unit": "iter/sec",
            "range": "stddev: 0.000003243937992398547",
            "extra": "mean: 70.4608765412851 usec\nrounds: 13057"
          },
          {
            "name": "benchmarks/test_box.py::test_box_wrap[reg-1k]",
            "value": 5232.039162624891,
            "unit": "iter/sec",
            "range": "stddev: 0.000022258299000642434",
            "extra": "mean: 191.13006782202763 usec\nrounds: 3981"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_create[reg-1k]",
            "value": 79034.57078523746,
            "unit": "iter/sec",
            "range": "stddev: 0.000002242222304274114",
            "extra": "mean: 12.652690968833424 usec\nrounds: 15856"
          },
          {
            "name": "benchmarks/test_frame.py::test_frame_block_access[reg-1k]",
            "value": 314261.92668753245,
            "unit": "iter/sec",
            "range": "stddev: 9.058092899446012e-7",
            "extra": "mean: 3.182059024904694 usec\nrounds: 85896"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo[reg-1k]",
            "value": 1428.3110402071434,
            "unit": "iter/sec",
            "range": "stddev: 0.00003558329513404992",
            "extra": "mean: 700.1276135588598 usec\nrounds: 885"
          },
          {
            "name": "benchmarks/test_topology.py::test_get_topo_distances[reg-1k]",
            "value": 189.41616017086926,
            "unit": "iter/sec",
            "range": "stddev: 0.0032586475745531684",
            "extra": "mean: 5.279380592964803 msec\nrounds: 199"
          }
        ]
      }
    ]
  }
}