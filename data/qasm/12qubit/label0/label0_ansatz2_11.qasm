OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(1.5661999437438865) q[0];
rz(-3.140737866065367) q[0];
ry(-1.5707148617890763) q[1];
rz(-3.1411225561196687) q[1];
ry(-1.5718000342121696) q[2];
rz(3.137092650088126) q[2];
ry(-1.5707379511376736) q[3];
rz(-0.08094549131673918) q[3];
ry(-1.5696524131164302) q[4];
rz(2.095456406823202) q[4];
ry(-1.5728877690300118) q[5];
rz(2.525499191949074) q[5];
ry(-1.5709064762558371) q[6];
rz(-3.1410995064804994) q[6];
ry(-0.8175395566762382) q[7];
rz(0.7271015914851173) q[7];
ry(0.0018981539909436894) q[8];
rz(-0.5238687194931747) q[8];
ry(-3.06122878050131) q[9];
rz(-1.7175192740941256) q[9];
ry(1.5045702507420435) q[10];
rz(-0.4594137997347722) q[10];
ry(-1.5728468301801184) q[11];
rz(0.002160467337172349) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-0.46497599943170975) q[0];
rz(-0.7896208292108541) q[0];
ry(2.835796381057124) q[1];
rz(0.41818635041431435) q[1];
ry(-0.20047692971359865) q[2];
rz(1.6125745428160354) q[2];
ry(-0.00029549753659751396) q[3];
rz(-0.049904423572053595) q[3];
ry(3.1415462188196575) q[4];
rz(-1.6646790320329217) q[4];
ry(2.7625897887162828e-05) q[5];
rz(0.1812383211001647) q[5];
ry(-2.686871683575087) q[6];
rz(-1.5722189955125918) q[6];
ry(0.03508319050833375) q[7];
rz(2.298745081566397) q[7];
ry(-0.0003915460417225167) q[8];
rz(-0.4890855897523997) q[8];
ry(3.0995190672610398) q[9];
rz(2.3388491798979794) q[9];
ry(-0.5479749944056409) q[10];
rz(1.9883237573532437) q[10];
ry(2.9424149183982644) q[11];
rz(1.575701903892217) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-3.135569144490269) q[0];
rz(2.845629450538821) q[0];
ry(-3.1413176941660663) q[1];
rz(0.15215270327880653) q[1];
ry(-0.02798101573602185) q[2];
rz(1.4291166633099288) q[2];
ry(3.141461368702287) q[3];
rz(0.1944678068679977) q[3];
ry(-0.0019022197727966985) q[4];
rz(-0.9831071361329453) q[4];
ry(0.0016581493044487416) q[5];
rz(2.8083722512329015) q[5];
ry(-0.798903349752214) q[6];
rz(-0.006122023630010353) q[6];
ry(-3.1414939449880754) q[7];
rz(1.1693116117605502) q[7];
ry(0.0006767755638490414) q[8];
rz(1.2665624594438443) q[8];
ry(-1.5402495518522192) q[9];
rz(-2.94818991901216) q[9];
ry(-2.6440769876504824) q[10];
rz(-0.865086626382708) q[10];
ry(2.455909709108033) q[11];
rz(-1.5324905574923913) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(0.000438374195147695) q[0];
rz(1.9983373694023447) q[0];
ry(3.135972750435003) q[1];
rz(1.3099186814531592) q[1];
ry(-0.07146531138127798) q[2];
rz(1.6821924919510227) q[2];
ry(-0.9174903134575583) q[3];
rz(-1.7709465656724745) q[3];
ry(1.600939574970714) q[4];
rz(-0.3179905421264752) q[4];
ry(-0.374110522495017) q[5];
rz(2.3083061709128976) q[5];
ry(1.5779535263601083) q[6];
rz(1.8571948536157987) q[6];
ry(2.904785540797262) q[7];
rz(2.4355253062153244) q[7];
ry(1.5722530411292972) q[8];
rz(0.06458928973393974) q[8];
ry(1.4602280435032338) q[9];
rz(-1.097129241847627) q[9];
ry(3.012634231276548) q[10];
rz(-1.3486127532485066) q[10];
ry(3.109821423307416) q[11];
rz(2.9970166392493516) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(0.00011568323309029915) q[0];
rz(-2.490929232983107) q[0];
ry(-0.085552872821705) q[1];
rz(1.0017575348350156) q[1];
ry(-3.009852187891675) q[2];
rz(-1.5652518462652143) q[2];
ry(1.565146924826581) q[3];
rz(0.7011579822254372) q[3];
ry(-1.5505934668170687) q[4];
rz(3.1251238960343524) q[4];
ry(1.5350856664740957) q[5];
rz(1.9786078299125203) q[5];
ry(-3.1414956034043526) q[6];
rz(-0.03981826225318495) q[6];
ry(-3.1404467309940975) q[7];
rz(-0.7539131602688655) q[7];
ry(3.127923912612013) q[8];
rz(1.6187830951705404) q[8];
ry(0.00022252401252931264) q[9];
rz(0.9100944098199504) q[9];
ry(-0.00036861570663759835) q[10];
rz(-2.285635713373512) q[10];
ry(3.141399503626003) q[11];
rz(2.962734191117927) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-0.7070936774849679) q[0];
rz(1.570615586581446) q[0];
ry(-0.0021447916699450786) q[1];
rz(-1.0043787115396663) q[1];
ry(1.1574849617227774) q[2];
rz(-0.7627526889310374) q[2];
ry(3.1297258169231887) q[3];
rz(0.5176519547257505) q[3];
ry(-1.5261866310299652) q[4];
rz(-1.4636843706593543) q[4];
ry(-0.017167355092727377) q[5];
rz(-0.35884568232109176) q[5];
ry(-0.014240856793704458) q[6];
rz(-0.5231181735310984) q[6];
ry(-0.04823549821361084) q[7];
rz(0.2907702366234224) q[7];
ry(-3.110567161476925) q[8];
rz(-1.5893204878779468) q[8];
ry(-3.1415886617146174) q[9];
rz(-1.7550877050999123) q[9];
ry(-3.141414579701405) q[10];
rz(0.011585272895922037) q[10];
ry(0.4810217307436462) q[11];
rz(1.5709640913836902) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.8639873644848268) q[0];
rz(-2.5680086390915204) q[0];
ry(-0.03931870166664244) q[1];
rz(1.3932980846116134) q[1];
ry(-3.132011734320921) q[2];
rz(1.22317750302932) q[2];
ry(-0.013815793282589995) q[3];
rz(2.6157535093654922) q[3];
ry(-3.0927668724654835) q[4];
rz(-2.5136527067031227) q[4];
ry(2.8901761495041303) q[5];
rz(1.0900802080944958) q[5];
ry(3.0971241660244666) q[6];
rz(-1.3181123225809208) q[6];
ry(-0.3464448151001499) q[7];
rz(-2.615502328994534) q[7];
ry(-0.029076852433414402) q[8];
rz(-0.22340005950566028) q[8];
ry(-0.0017585281655686528) q[9];
rz(-1.585611341940855) q[9];
ry(-3.140999512843251) q[10];
rz(1.219525047349411) q[10];
ry(-0.5724884060692241) q[11];
rz(-1.571688100595212) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-3.141355071183434) q[0];
rz(-2.568753794605779) q[0];
ry(-3.12932365972703) q[1];
rz(-0.47355794056524947) q[1];
ry(-3.1382736855611495) q[2];
rz(-1.156118306244685) q[2];
ry(0.0022135843101578345) q[3];
rz(-0.9533989906391646) q[3];
ry(-3.1295225697164772) q[4];
rz(-2.07545410631391) q[4];
ry(0.017082692932266802) q[5];
rz(-0.9668083146909134) q[5];
ry(1.5733753057840634) q[6];
rz(-0.007597536949539387) q[6];
ry(0.029226815921551498) q[7];
rz(-1.6640910316572022) q[7];
ry(-0.0010697250511197964) q[8];
rz(-2.914452196793722) q[8];
ry(-3.1382621722640383) q[9];
rz(-1.3433975984761897) q[9];
ry(-3.1414301907932347) q[10];
rz(1.2392060690733535) q[10];
ry(-1.6033970709221188) q[11];
rz(-1.8843916301468802) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-3.041680020844169) q[0];
rz(-0.5978738554911709) q[0];
ry(-0.2645368240396541) q[1];
rz(-2.833449101433423) q[1];
ry(-1.5679303802518545) q[2];
rz(-2.9968329957331297) q[2];
ry(3.141538628948663) q[3];
rz(-1.6905263100334018) q[3];
ry(3.141207722223829) q[4];
rz(-1.2414109898044572) q[4];
ry(1.5839026327061765) q[5];
rz(1.7460758539468095) q[5];
ry(1.5639781674491902) q[6];
rz(1.7661543539301183) q[6];
ry(0.0024120529104362763) q[7];
rz(-1.96557249213101) q[7];
ry(1.5717096147414689) q[8];
rz(2.8319682249701) q[8];
ry(3.14131725419967) q[9];
rz(-0.023234526919657876) q[9];
ry(3.141455743400173) q[10];
rz(1.7145003006439723) q[10];
ry(-3.1405385817396865) q[11];
rz(-1.8833455059033346) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-3.528588615635753e-06) q[0];
rz(-1.9616491782113568) q[0];
ry(-1.5764682179550467) q[1];
rz(-1.5704627217239997) q[1];
ry(1.5478786621407528) q[2];
rz(-1.557643096663601) q[2];
ry(-0.24556004729461028) q[3];
rz(-0.4655827349490427) q[3];
ry(-0.030988696618144296) q[4];
rz(-3.0878820606616304) q[4];
ry(-3.1377572986605435) q[5];
rz(0.9670384170789759) q[5];
ry(-0.0012777204244849366) q[6];
rz(1.2856734131623782) q[6];
ry(-3.1412030750697912) q[7];
rz(1.626825049396678) q[7];
ry(3.141070486246686) q[8];
rz(1.0786925044227162) q[8];
ry(3.1415576025918863) q[9];
rz(-2.029305334994854) q[9];
ry(3.1415360803658485) q[10];
rz(2.921424346524051) q[10];
ry(-1.5668610030759496) q[11];
rz(-1.5166218829676614) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(3.1415585679516336) q[0];
rz(-2.37742096369174) q[0];
ry(-1.5559018309890051) q[1];
rz(3.1292849669558596) q[1];
ry(-1.5706625615937213) q[2];
rz(-3.138987392912607) q[2];
ry(-0.005061306163599787) q[3];
rz(-1.1706682881790949) q[3];
ry(-3.137578620734347) q[4];
rz(1.972822015539327) q[4];
ry(3.1404194265679948) q[5];
rz(0.7912908453040064) q[5];
ry(-0.006086949470515169) q[6];
rz(-1.7626215187192) q[6];
ry(-0.08995960448211186) q[7];
rz(1.2111214429028145) q[7];
ry(0.036324004392736586) q[8];
rz(-1.621671740454805) q[8];
ry(0.013289560635415755) q[9];
rz(-3.101014709111853) q[9];
ry(-3.1355737948319926) q[10];
rz(2.309570499747389) q[10];
ry(-0.005324044955118623) q[11];
rz(1.51828111085661) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(-2.8493192047766476) q[0];
rz(-1.0692352399884832) q[0];
ry(1.5775014088378485) q[1];
rz(-1.9283591686276333) q[1];
ry(1.572819961010969) q[2];
rz(-0.020082654408563272) q[2];
ry(3.140676417877939) q[3];
rz(1.9228068097332904) q[3];
ry(3.1415890207498527) q[4];
rz(2.179071027121811) q[4];
ry(1.5720330397088649) q[5];
rz(-2.383969588050908) q[5];
ry(0.0005469910602928178) q[6];
rz(1.1637379980412923) q[6];
ry(5.359180230257721e-06) q[7];
rz(-1.1773695061148095) q[7];
ry(0.00028372776353317213) q[8];
rz(-1.4210163555843467) q[8];
ry(3.1371614467770286) q[9];
rz(-1.4031134851458278) q[9];
ry(-0.0005472087432261407) q[10];
rz(1.4430141203586677) q[10];
ry(1.1247566036595096) q[11];
rz(1.599724094269625) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(0.27235896566491924) q[0];
rz(-0.22293080589962067) q[0];
ry(-0.022713908475960665) q[1];
rz(-2.8079859593967855) q[1];
ry(0.002293064626376129) q[2];
rz(-0.33464583702460166) q[2];
ry(-2.676211284491359e-05) q[3];
rz(2.835053413737088) q[3];
ry(-3.1414902055431515) q[4];
rz(-2.6593250400447133) q[4];
ry(0.00026830924522217774) q[5];
rz(1.887865746301177) q[5];
ry(3.1415221482551474) q[6];
rz(1.466296971095886) q[6];
ry(-3.141551560250877) q[7];
rz(2.7971374298834273) q[7];
ry(3.1415663992465954) q[8];
rz(-2.9658090194946376) q[8];
ry(-0.0006378575108505522) q[9];
rz(-1.802023318243805) q[9];
ry(0.00043407590747257524) q[10];
rz(-1.4949225969787783) q[10];
ry(-0.05691833158224568) q[11];
rz(-1.6003366425429277) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(1.8880975443895407) q[0];
rz(0.112077831849807) q[0];
ry(-1.5590620846111563) q[1];
rz(-3.1335714845548304) q[1];
ry(-0.0019399880029506988) q[2];
rz(1.922405095323322) q[2];
ry(-3.1408101232547265) q[3];
rz(0.13957139497268192) q[3];
ry(-3.1413583314000646) q[4];
rz(0.4377136231052932) q[4];
ry(4.7821476547441836e-05) q[5];
rz(-2.624790376435485) q[5];
ry(-3.1411010853213006) q[6];
rz(1.5701586161399235) q[6];
ry(-0.00015062700731416356) q[7];
rz(0.3962116427376054) q[7];
ry(3.1411466437484457) q[8];
rz(0.13626833755396353) q[8];
ry(-0.00436592505478739) q[9];
rz(-2.573623325648611) q[9];
ry(-3.1409043008120054) q[10];
rz(1.7993381584563195) q[10];
ry(1.6354536958538612) q[11];
rz(-1.570381230882349) q[11];
cz q[0],q[1];
cz q[0],q[2];
cz q[0],q[3];
cz q[0],q[4];
cz q[0],q[5];
cz q[0],q[6];
cz q[0],q[7];
cz q[0],q[8];
cz q[0],q[9];
cz q[0],q[10];
cz q[0],q[11];
cz q[1],q[2];
cz q[1],q[3];
cz q[1],q[4];
cz q[1],q[5];
cz q[1],q[6];
cz q[1],q[7];
cz q[1],q[8];
cz q[1],q[9];
cz q[1],q[10];
cz q[1],q[11];
cz q[2],q[3];
cz q[2],q[4];
cz q[2],q[5];
cz q[2],q[6];
cz q[2],q[7];
cz q[2],q[8];
cz q[2],q[9];
cz q[2],q[10];
cz q[2],q[11];
cz q[3],q[4];
cz q[3],q[5];
cz q[3],q[6];
cz q[3],q[7];
cz q[3],q[8];
cz q[3],q[9];
cz q[3],q[10];
cz q[3],q[11];
cz q[4],q[5];
cz q[4],q[6];
cz q[4],q[7];
cz q[4],q[8];
cz q[4],q[9];
cz q[4],q[10];
cz q[4],q[11];
cz q[5],q[6];
cz q[5],q[7];
cz q[5],q[8];
cz q[5],q[9];
cz q[5],q[10];
cz q[5],q[11];
cz q[6],q[7];
cz q[6],q[8];
cz q[6],q[9];
cz q[6],q[10];
cz q[6],q[11];
cz q[7],q[8];
cz q[7],q[9];
cz q[7],q[10];
cz q[7],q[11];
cz q[8],q[9];
cz q[8],q[10];
cz q[8],q[11];
cz q[9],q[10];
cz q[9],q[11];
cz q[10],q[11];
ry(3.1374764913232687) q[0];
rz(2.941438074456056) q[0];
ry(1.5726696071123252) q[1];
rz(1.48986392350026) q[1];
ry(1.5528943503349515) q[2];
rz(-1.5707968309215659) q[2];
ry(1.5873476223109622) q[3];
rz(-1.5704446821025184) q[3];
ry(-1.5544404541302175) q[4];
rz(1.5839554169918326) q[4];
ry(-1.4977805524365717) q[5];
rz(-3.127077387970959) q[5];
ry(1.6027244087781716) q[6];
rz(1.6044738631505089) q[6];
ry(0.7210259934084259) q[7];
rz(0.041100056920190475) q[7];
ry(1.5822844742516124) q[8];
rz(0.24479177005325595) q[8];
ry(0.04953405568356409) q[9];
rz(-0.7133788586216187) q[9];
ry(0.08465329595860105) q[10];
rz(1.2102369826998522) q[10];
ry(0.482809129327467) q[11];
rz(1.5718021743652235) q[11];