import numpy as np
import matplotlib.pyplot as pyplot

ranks = ["0-100", "101, 200", "201, 300", "301,400", "401, 500", "501, 600", "601, 700", "701, 800", "801, 1200", "1201, 1300", "1301, 1400", "1401, 1500", "1501, 1600", "1601, 1700", "1701, 1800", "1801, 1900", "1901, 2000", "1950-2050"]
'''accur_max = [0.5336564351103931, 0.5325794291868605, 0.5336564351103931, 0.5325794291868605, 0.5347334410339257, 0.5325794291868605, 0.5312331717824448, 0.5301561658589122, 0.531502423263328, 0.5301561658589122, 0.5309639203015617, 0.5301561658589122]
accur1 = [0.9553571428571429, 0.9375, 0.9553571428571429, 0.9375, 0.9553571428571429, 0.9375, 0.9375, 0.9196428571428571, 0.9375, 0.9196428571428571, 0.9285714285714286, 0.9196428571428571]
accur2 = [0.8434065934065934, 0.8021978021978022, 0.8626373626373627, 0.8021978021978022, 0.8626373626373627, 0.8021978021978022, 0.804945054945055, 0.7747252747252747, 0.7994505494505495, 0.771978021978022, 0.7884615384615384, 0.771978021978022]
accur3 = [0.6209677419354839, 0.6370967741935484, 0.6129032258064516, 0.6370967741935484, 0.5967741935483871, 0.6370967741935484, 0.6129032258064516, 0.6693548387096774, 0.6330645161290323, 0.6653225806451613, 0.6048387096774194, 0.6653225806451613]
accur4 = [0.8204787234042553, 0.7460106382978723, 0.824468085106383, 0.7460106382978723, 0.8191489361702128, 0.7460106382978723, 0.8789893617021277, 0.8643617021276596, 0.875, 0.8643617021276596, 0.8776595744680851, 0.8643617021276596]
accur5 = [0.7055961070559611, 0.732360097323601, 0.7116788321167883, 0.732360097323601, 0.7055961070559611, 0.732360097323601, 0.6873479318734793, 0.7262773722627737, 0.6885644768856448, 0.7287104622871047, 0.6861313868613139, 0.7287104622871047]
accur6 = [0.4985549132947977, 0.49710982658959535, 0.4985549132947977, 0.49710982658959535, 0.4956647398843931, 0.49710982658959535, 0.5289017341040463, 0.5361271676300579, 0.5260115606936416, 0.5361271676300579, 0.5303468208092486, 0.5361271676300579]
accur7 = [0.5012406947890818, 0.49255583126550867, 0.5012406947890818, 0.49255583126550867, 0.5049627791563276, 0.49255583126550867, 0.5198511166253101, 0.5173697270471465, 0.5173697270471465, 0.5173697270471465, 0.5210918114143921, 0.5173697270471465]
accur_mid = [0.9789473684210527, 0.9578947368421052, 0.9789473684210527, 0.9578947368421052, 0.9789473684210527, 0.9578947368421052, 0.9894736842105263, 0.9578947368421052, 0.9894736842105263, 0.9578947368421052, 0.9894736842105263, 0.9578947368421052]
accur8 = [0.5213730569948186, 0.5362694300518135, 0.5213730569948186, 0.5362694300518135, 0.5239637305699482, 0.5362694300518135, 0.5453367875647669, 0.5382124352331606, 0.5446891191709845, 0.538860103626943, 0.5479274611398963, 0.538860103626943]
accur9 = [0.6213842975206612, 0.6260330578512396, 0.6244834710743802, 0.6260330578512396, 0.625, 0.6260330578512396, 0.6849173553719008, 0.6988636363636364, 0.6947314049586777, 0.6978305785123967, 0.6921487603305785, 0.6978305785123967]
accur10 = [0.6025974025974026, 0.5746753246753247, 0.6035714285714285, 0.5746753246753247, 0.6055194805194806, 0.5746753246753247, 0.6152597402597403, 0.6100649350649351, 0.6201298701298701, 0.6100649350649351, 0.6185064935064936, 0.6100649350649351]
accur11 = [0.6051797040169133, 0.5998942917547568, 0.6057082452431289, 0.5998942917547568, 0.6009513742071881, 0.5998942917547568, 0.6051797040169133, 0.5993657505285412, 0.6088794926004228, 0.5998942917547568, 0.607293868921776, 0.6004228329809725]
accur12 = [0.6028901734104046, 0.6063583815028901, 0.6040462427745664, 0.6063583815028901, 0.599421965317919, 0.6063583815028901, 0.6184971098265896, 0.615606936416185, 0.6150289017341041, 0.6150289017341041, 0.6196531791907515, 0.6150289017341041]
accur13 = [0.5791873963515755, 0.5698590381426202, 0.5789800995024875, 0.5698590381426202, 0.5766998341625207, 0.5698590381426202, 0.5968076285240465, 0.5986733001658375, 0.5978441127694859, 0.5984660033167496, 0.5959784411276948, 0.5982587064676617]
accur14 = [0.5354477611940298, 0.530223880597015, 0.5365671641791044, 0.530223880597015, 0.5358208955223881, 0.530223880597015, 0.5496268656716418, 0.567910447761194, 0.5511194029850747, 0.5675373134328359, 0.5507462686567164, 0.5675373134328359]
accur15 = [0.5672514619883041, 0.5542763157894737, 0.5672514619883041, 0.5542763157894737, 0.5656067251461988, 0.5542763157894737, 0.5705409356725146, 0.5782163742690059, 0.5720029239766082, 0.5782163742690059, 0.5712719298245614, 0.5782163742690059]
accur_min = [0.6293969849246231, 0.6331658291457286, 0.6293969849246231, 0.6331658291457286, 0.628140703517588, 0.6331658291457286, 0.6457286432160804, 0.6042713567839196, 0.6482412060301508, 0.6042713567839196, 0.6457286432160804, 0.6042713567839196]
'''




accur_min = [0.4305555555555556, 0.4583333333333333, 0.4305555555555556, 0.4583333333333333, 0.4305555555555556, 0.4583333333333333, 0.4583333333333333, 0.4583333333333333, 0.4444444444444444, 0.4583333333333333, 0.4583333333333333, 0.4583333333333333]
accur1 = [0.9553571428571429, 0.9375, 0.9553571428571429, 0.9375, 0.9553571428571429, 0.9375, 0.9375, 0.9196428571428571, 0.9375, 0.9196428571428571, 0.9285714285714286, 0.9196428571428571]
accur2 = [0.8434065934065934, 0.8021978021978022, 0.8626373626373627, 0.8021978021978022, 0.8626373626373627, 0.8021978021978022, 0.804945054945055, 0.7747252747252747, 0.7994505494505495, 0.771978021978022, 0.7884615384615384, 0.771978021978022]
accur3 = [0.6209677419354839, 0.6370967741935484, 0.6129032258064516, 0.6370967741935484, 0.5967741935483871, 0.6370967741935484, 0.6129032258064516, 0.6693548387096774, 0.6330645161290323, 0.6653225806451613, 0.6048387096774194, 0.6653225806451613]
accur4 = [0.8204787234042553, 0.7460106382978723, 0.824468085106383, 0.7460106382978723, 0.8191489361702128, 0.7460106382978723, 0.8789893617021277, 0.8643617021276596, 0.875, 0.8643617021276596, 0.8776595744680851, 0.8643617021276596]
accur5 = [0.7055961070559611, 0.732360097323601, 0.7116788321167883, 0.732360097323601, 0.7055961070559611, 0.732360097323601, 0.6873479318734793, 0.7262773722627737, 0.6885644768856448, 0.7287104622871047, 0.6861313868613139, 0.7287104622871047]
accur6 = [0.4985549132947977, 0.49710982658959535, 0.4985549132947977, 0.49710982658959535, 0.4956647398843931, 0.49710982658959535, 0.5289017341040463, 0.5361271676300579, 0.5260115606936416, 0.5361271676300579, 0.5303468208092486, 0.5361271676300579]
accur7 = [0.5012406947890818, 0.49255583126550867, 0.5012406947890818, 0.49255583126550867, 0.5049627791563276, 0.49255583126550867, 0.5198511166253101, 0.5173697270471465, 0.5173697270471465, 0.5173697270471465, 0.5210918114143921, 0.5173697270471465]
accur8 = [0.5213730569948186, 0.5362694300518135, 0.5213730569948186, 0.5362694300518135, 0.5239637305699482, 0.5362694300518135, 0.5453367875647669, 0.5382124352331606, 0.5446891191709845, 0.538860103626943, 0.5479274611398963, 0.538860103626943]
accur9 = [0.6213842975206612, 0.6260330578512396, 0.6244834710743802, 0.6260330578512396, 0.625, 0.6260330578512396, 0.6849173553719008, 0.6988636363636364, 0.6947314049586777, 0.6978305785123967, 0.6921487603305785, 0.6978305785123967]
accur10 = [0.6025974025974026, 0.5746753246753247, 0.6035714285714285, 0.5746753246753247, 0.6055194805194806, 0.5746753246753247, 0.6152597402597403, 0.6100649350649351, 0.6201298701298701, 0.6100649350649351, 0.6185064935064936, 0.6100649350649351]
accur11 = [0.6051797040169133, 0.5998942917547568, 0.6057082452431289, 0.5998942917547568, 0.6009513742071881, 0.5998942917547568, 0.6051797040169133, 0.5993657505285412, 0.6088794926004228, 0.5998942917547568, 0.607293868921776, 0.6004228329809725]
accur12 = [0.6028901734104046, 0.6063583815028901, 0.6040462427745664, 0.6063583815028901, 0.599421965317919, 0.6063583815028901, 0.6184971098265896, 0.615606936416185, 0.6150289017341041, 0.6150289017341041, 0.6196531791907515, 0.6150289017341041]
accur13 = [0.5791873963515755, 0.5698590381426202, 0.5789800995024875, 0.5698590381426202, 0.5766998341625207, 0.5698590381426202, 0.5968076285240465, 0.5986733001658375, 0.5978441127694859, 0.5984660033167496, 0.5959784411276948, 0.5982587064676617]
accur14 = [0.5354477611940298, 0.530223880597015, 0.5365671641791044, 0.530223880597015, 0.5358208955223881, 0.530223880597015, 0.5496268656716418, 0.567910447761194, 0.5511194029850747, 0.5675373134328359, 0.5507462686567164, 0.5675373134328359]
accur15 = [0.5672514619883041, 0.5542763157894737, 0.5672514619883041, 0.5542763157894737, 0.5656067251461988, 0.5542763157894737, 0.5705409356725146, 0.5782163742690059, 0.5720029239766082, 0.5782163742690059, 0.5712719298245614, 0.5782163742690059]
accur_max = [0.5522088353413654, 0.5507028112449799, 0.5522088353413654, 0.5507028112449799, 0.5512048192771084, 0.5507028112449799, 0.5461847389558233, 0.5471887550200804, 0.5486947791164659, 0.5471887550200804, 0.5456827309236948, 0.5471887550200804]





l = [accur_min, accur1, accur2, accur3, accur4, accur5, accur6, accur7, accur8, accur9, accur10, accur11, accur12, accur13, accur14, accur15, accur_max]

pyplot.boxplot(l)
pyplot.savefig("plot_ranks.png")