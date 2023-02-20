OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.0028466472889847006) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0455060303764245) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.024945493068754106) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.04348651537468914) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.08391821825317078) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.003602367202020655) q[3];
cx q[2],q[3];
rx(-0.03512241026474285) q[0];
rz(-0.07483645471903776) q[0];
rx(-0.03087996714562146) q[1];
rz(-0.1457295765598741) q[1];
rx(-0.07606736032417431) q[2];
rz(-0.11083256824422144) q[2];
rx(-0.06983619409074542) q[3];
rz(-0.017625608860444664) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0010998608102353166) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.055997592486985374) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.0757465542055401) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.004411214176324021) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.09415041128484136) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.14788934564168327) q[3];
cx q[2],q[3];
rx(-0.04416415810421982) q[0];
rz(-0.06543371852010814) q[0];
rx(-0.061646888569985396) q[1];
rz(-0.14450592004210555) q[1];
rx(0.04534847378130183) q[2];
rz(-0.1473774717283156) q[2];
rx(-0.1101251319428788) q[3];
rz(-0.030247453824201007) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.09026798615630262) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0014950683385529418) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.11691294298094984) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.08291474034492154) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.09983462540180818) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.16706938567728186) q[3];
cx q[2],q[3];
rx(-0.0330126086854327) q[0];
rz(-0.025894221176274275) q[0];
rx(-0.016908228006452895) q[1];
rz(-0.1719491081436482) q[1];
rx(0.04884261133542458) q[2];
rz(-0.17690910864730658) q[2];
rx(-0.14723573247187102) q[3];
rz(-0.047934437736954126) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.13759004864966307) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.028755332950111345) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.10556689801885681) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.041384573361962815) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1403667128491726) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.1303622191824292) q[3];
cx q[2],q[3];
rx(-0.14034564265429691) q[0];
rz(0.009486655612809904) q[0];
rx(-0.06388481628996118) q[1];
rz(-0.16371098096169423) q[1];
rx(-0.03653568861184366) q[2];
rz(-0.23580263646581973) q[2];
rx(-0.18470613508650247) q[3];
rz(-0.08327008084331051) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1639176509413412) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.03738443373243123) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.09827013257681466) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.09072135757017495) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.06733342270205162) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.12903975926600017) q[3];
cx q[2],q[3];
rx(-0.1503312778970544) q[0];
rz(0.00192506295093082) q[0];
rx(-0.09963483413553902) q[1];
rz(-0.20135910048764277) q[1];
rx(-0.07174801587828361) q[2];
rz(-0.20768474076038193) q[2];
rx(-0.16603957323719473) q[3];
rz(-0.01672972201573967) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.22028262473793656) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.027607936537956138) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.017993689123905046) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.2569810317772635) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.0434163810606907) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.1882174599648977) q[3];
cx q[2],q[3];
rx(-0.14614441700203637) q[0];
rz(-0.04427162812385652) q[0];
rx(-0.08684917321871374) q[1];
rz(-0.14139843905811372) q[1];
rx(-0.013096230729130024) q[2];
rz(-0.1840100538878219) q[2];
rx(-0.22216117204374286) q[3];
rz(0.014651700607440692) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.213555762302496) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.03249936770657201) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.05112101439221407) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.2921352475697859) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.00805035447897855) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.17674780873688528) q[3];
cx q[2],q[3];
rx(-0.15015109361481133) q[0];
rz(0.0028376196776888135) q[0];
rx(-0.11622582795687514) q[1];
rz(-0.2045299541983179) q[1];
rx(0.09241000628988744) q[2];
rz(-0.2173797278232189) q[2];
rx(-0.24464806640686493) q[3];
rz(0.037165784161188646) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.23241301148075977) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.015768953989512047) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.002503129739736052) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.18667917909752116) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.035045817399893944) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.20142003943299552) q[3];
cx q[2],q[3];
rx(-0.23132995270396192) q[0];
rz(-0.04052156348596484) q[0];
rx(-0.14348109064252162) q[1];
rz(-0.24912312338510073) q[1];
rx(-0.014300552958056049) q[2];
rz(-0.14943794615534148) q[2];
rx(-0.29103019898386445) q[3];
rz(0.017300579520407703) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1567705967437168) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.03642164355827387) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.05540929034829908) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.126825873540447) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.03815058296445551) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.11344378132122276) q[3];
cx q[2],q[3];
rx(-0.22985930700309762) q[0];
rz(0.011169693551471611) q[0];
rx(-0.10792248416853983) q[1];
rz(-0.3178524645101553) q[1];
rx(-0.06806029242762854) q[2];
rz(-0.08589738931876523) q[2];
rx(-0.2832945269224708) q[3];
rz(-0.022137812191256018) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.17030213648459278) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.031514378755763864) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.03372498398170597) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1869091017282614) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.01196691857313116) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.08288638846531826) q[3];
cx q[2],q[3];
rx(-0.2905020609753848) q[0];
rz(-0.00830754094387162) q[0];
rx(-0.03673971965875854) q[1];
rz(-0.3371746956610401) q[1];
rx(0.016583648541404993) q[2];
rz(-0.034707148885505405) q[2];
rx(-0.28612596915815275) q[3];
rz(-0.05013621509004714) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1930985758426207) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.04698099700842269) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.020658196777628698) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.15033342404309985) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1333663348213092) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.010928079971769962) q[3];
cx q[2],q[3];
rx(-0.28229141580645545) q[0];
rz(-0.020168472569975802) q[0];
rx(-0.015059208665988616) q[1];
rz(-0.2864982638774043) q[1];
rx(-0.01168782305064355) q[2];
rz(-0.07536294244280359) q[2];
rx(-0.309152007837314) q[3];
rz(-0.06034638666515291) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.20080403760623883) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.02692323588500895) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.05958386950343041) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.2579842032491499) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.19685621940066203) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.02439052363320757) q[3];
cx q[2],q[3];
rx(-0.2843834338717767) q[0];
rz(0.01185357655495934) q[0];
rx(-0.02592104619031184) q[1];
rz(-0.25266667916626817) q[1];
rx(-0.02172792046025791) q[2];
rz(-0.12226921846340541) q[2];
rx(-0.21487986231231804) q[3];
rz(-0.046424104300068464) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.16509730636622444) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.037908328395847485) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.006369347758644334) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.2590395024844051) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.14091671371958747) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.002750531480341391) q[3];
cx q[2],q[3];
rx(-0.23298565341412122) q[0];
rz(0.07814445361557064) q[0];
rx(-0.053480258420590086) q[1];
rz(-0.18236837226440478) q[1];
rx(0.069297517112507) q[2];
rz(-0.1563354388501822) q[2];
rx(-0.22038007804573234) q[3];
rz(-0.04784817768745831) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.12202390415819397) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.059618411060593714) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.042957658672315775) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.2559712834963242) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.21903511029838021) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.08609051160220657) q[3];
cx q[2],q[3];
rx(-0.29053213597652494) q[0];
rz(-0.009460437283678408) q[0];
rx(-0.13217884953324766) q[1];
rz(-0.12324139420153586) q[1];
rx(-0.01499960706558076) q[2];
rz(-0.18307283512280392) q[2];
rx(-0.19991021121929764) q[3];
rz(-0.05136156311967844) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.15811828078725876) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.11510973126872125) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.027221116603330712) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.20678362872674003) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.0998012183955472) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.08690113585336062) q[3];
cx q[2],q[3];
rx(-0.3186354068810729) q[0];
rz(-0.036207651429814296) q[0];
rx(-0.04292861179650856) q[1];
rz(-0.08548428991590926) q[1];
rx(-0.02539889600134414) q[2];
rz(-0.17884623804332747) q[2];
rx(-0.1843616618244644) q[3];
rz(-0.044975267954307965) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.12274335725758423) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.005921260592495803) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.03625286378882122) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1559155046287548) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.04642868367314472) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.02215802320192153) q[3];
cx q[2],q[3];
rx(-0.29765788117377046) q[0];
rz(0.05117115950214316) q[0];
rx(-0.02721143107775628) q[1];
rz(-0.05910720589813992) q[1];
rx(0.006302679422287001) q[2];
rz(-0.08235334752237859) q[2];
rx(-0.15230766942504192) q[3];
rz(-0.08085308428361081) q[3];