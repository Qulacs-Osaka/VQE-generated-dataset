OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-2.3140312097385016) q[0];
rz(2.041181626671758) q[0];
ry(2.0567819896847053) q[1];
rz(1.2149226586068878) q[1];
ry(-0.0028571226623676496) q[2];
rz(0.31837272840453357) q[2];
ry(3.140964914119565) q[3];
rz(-2.253664837362099) q[3];
ry(1.5542196681340328) q[4];
rz(-2.9737395580951924) q[4];
ry(-1.5769498573904073) q[5];
rz(-3.0586349176054726) q[5];
ry(0.00958046295901066) q[6];
rz(1.0239200843921932) q[6];
ry(-7.851908692467046e-05) q[7];
rz(-1.3899135017306383) q[7];
ry(1.520870152271951) q[8];
rz(1.5702591383524993) q[8];
ry(-0.04675229010017372) q[9];
rz(-0.1198907623399334) q[9];
ry(1.0399882261740574) q[10];
rz(-0.043093087465769564) q[10];
ry(2.5732290558603697) q[11];
rz(-0.46852931620735355) q[11];
ry(-0.7746594184141102) q[12];
rz(-3.1239362420303673) q[12];
ry(-1.407812882286481) q[13];
rz(-1.0299184803953505) q[13];
ry(0.6340442904550345) q[14];
rz(-2.6665813135137224) q[14];
ry(0.11028429314814847) q[15];
rz(-2.8589862413266762) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.5825598860685004) q[0];
rz(-0.13053830087218088) q[0];
ry(2.2104982355279033) q[1];
rz(-1.5621429889237) q[1];
ry(-3.1003706889972853) q[2];
rz(-1.620163628076739) q[2];
ry(3.10762831656529) q[3];
rz(2.967907222930272) q[3];
ry(-1.0133661818157516) q[4];
rz(1.0092912742490316) q[4];
ry(0.19152859818904347) q[5];
rz(-1.217968969691981) q[5];
ry(-1.8154785540802305) q[6];
rz(2.824520858811823) q[6];
ry(-1.559885597228126) q[7];
rz(1.392782880930538) q[7];
ry(-3.0906865116840945) q[8];
rz(0.813444379963331) q[8];
ry(-3.11484178069557) q[9];
rz(-0.3373082441943345) q[9];
ry(3.1414253303510025) q[10];
rz(-0.5540675707425363) q[10];
ry(0.0010703173119219753) q[11];
rz(-3.0861234316280615) q[11];
ry(0.1213363627919595) q[12];
rz(-2.677196332087078) q[12];
ry(0.4127113187184356) q[13];
rz(-1.3910128821063656) q[13];
ry(0.004062516456551537) q[14];
rz(2.572654204776646) q[14];
ry(-2.920324433384414) q[15];
rz(-2.789725527373707) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.9728073446420158) q[0];
rz(0.9593760346390957) q[0];
ry(-2.767084260473982) q[1];
rz(-2.667881253330187) q[1];
ry(0.04125060689825653) q[2];
rz(-2.70598912330077) q[2];
ry(-0.0037690745478133935) q[3];
rz(-1.621471997549337) q[3];
ry(-3.0516955005160056) q[4];
rz(-1.1575092799452076) q[4];
ry(2.789992123645974) q[5];
rz(1.672914052193241) q[5];
ry(3.1392993004207472) q[6];
rz(-2.0973161854342752) q[6];
ry(-0.012630615641850975) q[7];
rz(0.3020535841571207) q[7];
ry(0.4227017986738409) q[8];
rz(-1.9411236069905913) q[8];
ry(0.6507200124888983) q[9];
rz(-2.9498520458832727) q[9];
ry(0.7571550095315738) q[10];
rz(-3.0435675193399554) q[10];
ry(-2.293773078003883) q[11];
rz(1.4792462387403225) q[11];
ry(2.021710186609993) q[12];
rz(0.10278001930110213) q[12];
ry(-2.9379229579063404) q[13];
rz(-2.340343324130626) q[13];
ry(2.402439989955853) q[14];
rz(1.0829520090640186) q[14];
ry(-2.226854807058317) q[15];
rz(-1.5464732480944487) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.7168007971232138) q[0];
rz(1.2737373656384765) q[0];
ry(0.15390295502244328) q[1];
rz(-0.43696108504490816) q[1];
ry(0.008854452337279994) q[2];
rz(-2.9665095200427327) q[2];
ry(1.5926966324569776) q[3];
rz(1.200129637961497) q[3];
ry(0.02271832045589249) q[4];
rz(-2.6471275548754165) q[4];
ry(2.7177523740902814) q[5];
rz(1.1851954770993534) q[5];
ry(-0.06481765391377381) q[6];
rz(1.7598228798217335) q[6];
ry(-0.14358290645618957) q[7];
rz(-2.0581249216502076) q[7];
ry(3.1393766786309345) q[8];
rz(-1.053030875642216) q[8];
ry(-3.1243039406095576) q[9];
rz(-2.7738147100062434) q[9];
ry(1.0643178723822647e-05) q[10];
rz(-3.0052793717247264) q[10];
ry(-0.0023908315854531074) q[11];
rz(1.28077213540434) q[11];
ry(-2.250504899232332) q[12];
rz(2.9225727127230336) q[12];
ry(-0.9229325830583518) q[13];
rz(0.7331625756510078) q[13];
ry(-1.4339949315164722) q[14];
rz(1.9890199084077105) q[14];
ry(2.0023043527095714) q[15];
rz(-3.0508856244742484) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-3.0570433728883932) q[0];
rz(0.9176047980322721) q[0];
ry(0.3627998092724235) q[1];
rz(0.0022841093362853115) q[1];
ry(0.08147239270879725) q[2];
rz(1.5246007679909879) q[2];
ry(-2.770288397552786) q[3];
rz(-0.4162343034813656) q[3];
ry(3.137787627169439) q[4];
rz(-2.1944544614049377) q[4];
ry(-3.1386129634553868) q[5];
rz(-2.0797446453591037) q[5];
ry(0.026632278426368183) q[6];
rz(1.5808649361989824) q[6];
ry(1.2481145684665318) q[7];
rz(0.14664465984554304) q[7];
ry(1.9823796841196888) q[8];
rz(-2.442928518304176) q[8];
ry(-2.489598840215648) q[9];
rz(1.0980651426017578) q[9];
ry(-1.471528696006457) q[10];
rz(1.8836210184432112) q[10];
ry(-1.804802055903753) q[11];
rz(2.9419505857602912) q[11];
ry(2.800237629662103) q[12];
rz(1.7040019113160605) q[12];
ry(1.5733393509309732) q[13];
rz(2.7373118337004207) q[13];
ry(0.9484290103696189) q[14];
rz(-1.6753240513715266) q[14];
ry(3.0860380020741673) q[15];
rz(0.31475866877025016) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.885902007689852) q[0];
rz(-1.5525356519011835) q[0];
ry(0.043325197461420835) q[1];
rz(-0.34961349686042104) q[1];
ry(-3.1150640844641755) q[2];
rz(1.0870639941846567) q[2];
ry(-0.0568461612985776) q[3];
rz(0.6779630876149048) q[3];
ry(-2.8008158545925284) q[4];
rz(-0.04432936166073542) q[4];
ry(2.7876260186695165) q[5];
rz(3.1035642686458194) q[5];
ry(-1.5581734531808409) q[6];
rz(-1.092625079685209) q[6];
ry(1.5603005357958335) q[7];
rz(3.0600400392187543) q[7];
ry(0.09245012019211174) q[8];
rz(-2.334217970949551) q[8];
ry(1.5822017509182276) q[9];
rz(-0.02067644943070756) q[9];
ry(3.127864622280458) q[10];
rz(-2.2755587578244434) q[10];
ry(0.0338782172581256) q[11];
rz(-1.5626159368839687) q[11];
ry(-0.10852270389655125) q[12];
rz(-1.2262037060817477) q[12];
ry(-3.1088984338999484) q[13];
rz(0.39979083816282657) q[13];
ry(3.0492365779061137) q[14];
rz(-1.9928629348556122) q[14];
ry(-1.8071517955339402) q[15];
rz(2.827962275081236) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(2.435311803730613) q[0];
rz(-1.9056955841037344) q[0];
ry(-1.8175443287096629) q[1];
rz(-2.6060367588156277) q[1];
ry(-1.5487247680000968) q[2];
rz(-1.5740796647299933) q[2];
ry(0.09764628985025904) q[3];
rz(1.3124008946253936) q[3];
ry(1.5698824159488485) q[4];
rz(1.6806366984508785) q[4];
ry(1.570393203339953) q[5];
rz(1.6817936704489131) q[5];
ry(0.021287134419022153) q[6];
rz(-2.035451087956961) q[6];
ry(1.581559716438651) q[7];
rz(-1.5651965120814777) q[7];
ry(-0.0015921881372737321) q[8];
rz(-2.2088727741681726) q[8];
ry(-2.5416890140162294) q[9];
rz(-1.2569867885816886) q[9];
ry(1.5768735247930543) q[10];
rz(-1.5929724211303062) q[10];
ry(-1.579619954243793) q[11];
rz(-1.4970777538397175) q[11];
ry(2.2288158463860155) q[12];
rz(-1.178087514689459) q[12];
ry(-1.414613249546975) q[13];
rz(-0.2826733654813536) q[13];
ry(-2.741179675238268) q[14];
rz(-1.7874233282142202) q[14];
ry(2.2941835473250447) q[15];
rz(2.269283884767041) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-2.84785343288273) q[0];
rz(-0.06273521049103081) q[0];
ry(-1.2791354673820718) q[1];
rz(1.4247917657660016) q[1];
ry(1.3651655673982943) q[2];
rz(-1.5389505442254265) q[2];
ry(-1.5768182241536761) q[3];
rz(1.7532457896369409) q[3];
ry(-1.5723568330100022) q[4];
rz(1.572355387011732) q[4];
ry(1.569043832739708) q[5];
rz(6.387293045397491e-05) q[5];
ry(1.5681860408624202) q[6];
rz(0.005480932480869782) q[6];
ry(-1.5546431370963985) q[7];
rz(1.4878077172140491) q[7];
ry(-0.00736635326849296) q[8];
rz(-0.8909409910107438) q[8];
ry(-3.1382066402918993) q[9];
rz(-2.626688704947097) q[9];
ry(0.4680207182457341) q[10];
rz(0.013846355920542916) q[10];
ry(-0.0737961729133838) q[11];
rz(2.9568100719686474) q[11];
ry(0.8399675920234956) q[12];
rz(-3.124568393368378) q[12];
ry(0.6859864901688487) q[13];
rz(-1.921595320458945) q[13];
ry(-2.438817115328276) q[14];
rz(1.138873008067768) q[14];
ry(0.4244469860579292) q[15];
rz(2.880394819472008) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.2854461726352788) q[0];
rz(-2.526326602186136) q[0];
ry(-0.9087360583013178) q[1];
rz(-1.8071621927038664) q[1];
ry(1.5915682032634546) q[2];
rz(-2.1346778025378184) q[2];
ry(-1.5430547034160966) q[3];
rz(-1.783760770169277) q[3];
ry(-0.0011183655561594202) q[4];
rz(-3.126188823903203) q[4];
ry(1.580914923218999) q[5];
rz(0.23587385877731543) q[5];
ry(-1.4442965976329143) q[6];
rz(-3.098926882894243) q[6];
ry(-1.5820383438592762) q[7];
rz(1.6287717697891066) q[7];
ry(3.0990977441858365) q[8];
rz(-1.530004720337079) q[8];
ry(-0.065032108836518) q[9];
rz(-1.7291305384489022) q[9];
ry(2.6411341946462294) q[10];
rz(-0.8452172154180537) q[10];
ry(0.02970498588492695) q[11];
rz(0.10705695650539848) q[11];
ry(-0.11497862087931221) q[12];
rz(2.398765372506775) q[12];
ry(1.1256801662419544) q[13];
rz(2.4337230954418136) q[13];
ry(-1.6442605498873029) q[14];
rz(-3.1407096053180084) q[14];
ry(1.2509685774803156) q[15];
rz(-2.1925292936058716) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-0.29646472469568985) q[0];
rz(-3.0148670608829566) q[0];
ry(-3.0484055354177655) q[1];
rz(-0.8994436434092918) q[1];
ry(-0.012250994874220211) q[2];
rz(0.6263968903526854) q[2];
ry(0.019419822545059966) q[3];
rz(-2.570683461343941) q[3];
ry(3.101196853647007) q[4];
rz(0.0071542747508734905) q[4];
ry(-2.183486763255574) q[5];
rz(-2.782777761517175) q[5];
ry(1.5569029879159535) q[6];
rz(2.114560892745029) q[6];
ry(-1.5639645112895781) q[7];
rz(-1.249817636937757) q[7];
ry(-1.5665726368408546) q[8];
rz(-0.45460590180798344) q[8];
ry(-1.5381987988176098) q[9];
rz(3.061904751189372) q[9];
ry(0.007398197766278258) q[10];
rz(-2.298048137294902) q[10];
ry(3.1155466827292058) q[11];
rz(1.4044463138239527) q[11];
ry(-2.8200536665256624) q[12];
rz(-1.4063644099365933) q[12];
ry(-2.8477703377437633) q[13];
rz(-1.1326854154542072) q[13];
ry(2.4088695727046443) q[14];
rz(1.87449455673385) q[14];
ry(-0.6658392480091084) q[15];
rz(1.9016421942083719) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(2.9268086681692593) q[0];
rz(1.35477625007458) q[0];
ry(1.6231794871584564) q[1];
rz(1.271290611542029) q[1];
ry(3.1408215798424624) q[2];
rz(1.286551194697301) q[2];
ry(3.139538189938236) q[3];
rz(2.6712473850934546) q[3];
ry(0.4162056758716208) q[4];
rz(-2.451477602959783) q[4];
ry(3.0564381631994446) q[5];
rz(3.0804584142555096) q[5];
ry(-3.1342486553019753) q[6];
rz(2.147032941501009) q[6];
ry(0.008636908242424468) q[7];
rz(-1.6499893984927718) q[7];
ry(-0.014400914437266096) q[8];
rz(0.3929765159078258) q[8];
ry(-1.557981995571298) q[9];
rz(-1.627455015827278) q[9];
ry(1.5784801292124886) q[10];
rz(-1.6000733916216046) q[10];
ry(3.0845664355196556) q[11];
rz(-0.15407897065347126) q[11];
ry(-0.165884718633837) q[12];
rz(0.3197387730096945) q[12];
ry(-1.5647610520993938) q[13];
rz(1.6945601158829526) q[13];
ry(2.4919810432217213) q[14];
rz(2.3464218786207645) q[14];
ry(-2.4981881674955675) q[15];
rz(2.258549389781449) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.6867875891489712) q[0];
rz(1.4786409275868533) q[0];
ry(2.1861078021371982) q[1];
rz(-0.29195771591617437) q[1];
ry(0.006837833813538426) q[2];
rz(0.26172385860845027) q[2];
ry(3.1305247508466345) q[3];
rz(1.6974889750372841) q[3];
ry(3.139934147230164) q[4];
rz(0.6816804265540712) q[4];
ry(-2.1767526592495474) q[5];
rz(-1.5862570015877293) q[5];
ry(3.1402450480447914) q[6];
rz(-1.0022604092854186) q[6];
ry(-3.1390561374131103) q[7];
rz(2.345812805992683) q[7];
ry(2.943307777083533) q[8];
rz(3.073284116182757) q[8];
ry(-1.5657008195623956) q[9];
rz(-1.6018232776042656) q[9];
ry(1.569706184655847) q[10];
rz(1.5486490382249418) q[10];
ry(1.5710593642629425) q[11];
rz(-1.5728611752940216) q[11];
ry(-1.5682885396023627) q[12];
rz(-1.573085596203745) q[12];
ry(-1.5710317028697292) q[13];
rz(1.568161498433927) q[13];
ry(2.880977976933918) q[14];
rz(-2.9349808518229006) q[14];
ry(-0.7359391750206239) q[15];
rz(-2.0882288396579662) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.2988878826346009) q[0];
rz(-1.9532935526479738) q[0];
ry(2.653735419603829) q[1];
rz(1.7533711000196277) q[1];
ry(0.02497488911189016) q[2];
rz(1.7204541829548354) q[2];
ry(-0.0021912666910459677) q[3];
rz(-2.5336595516424203) q[3];
ry(-1.3250396426116224) q[4];
rz(-2.2261938741131155) q[4];
ry(1.5689427540665177) q[5];
rz(-2.8519307281657698) q[5];
ry(-3.1342025818494785) q[6];
rz(2.6533101387963525) q[6];
ry(-1.567254968994109) q[7];
rz(-0.897002657639854) q[7];
ry(1.5861512567523832) q[8];
rz(-1.9950589914737602) q[8];
ry(1.5546969630975331) q[9];
rz(-1.450376407921115) q[9];
ry(-0.035033176338683525) q[10];
rz(1.5879125679729613) q[10];
ry(1.7410354357361115) q[11];
rz(1.5437614087768008) q[11];
ry(-1.683764931665995) q[12];
rz(-0.039664659382546714) q[12];
ry(-1.4970342157456866) q[13];
rz(-0.007815089503601868) q[13];
ry(2.6668320129761796) q[14];
rz(-1.891437071353299) q[14];
ry(3.008545556796914) q[15];
rz(1.3926171046729907) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-3.033623890491159) q[0];
rz(-2.411420345409604) q[0];
ry(2.1201474400719116) q[1];
rz(-2.065049914188389) q[1];
ry(-2.600784255462602) q[2];
rz(-0.9491595072128363) q[2];
ry(-1.5972049030734743) q[3];
rz(2.264748445006604) q[3];
ry(1.5916732925245043) q[4];
rz(-3.140338087852578) q[4];
ry(3.1399249716068143) q[5];
rz(-0.6121513489461406) q[5];
ry(1.5583488433732082) q[6];
rz(-2.3478321402506532) q[6];
ry(-3.140798918553775) q[7];
rz(-2.5375772249430337) q[7];
ry(-3.139184307696844) q[8];
rz(-0.057866963765889956) q[8];
ry(0.008586800280149907) q[9];
rz(0.9637194536973368) q[9];
ry(-1.5765976503682033) q[10];
rz(1.6584585503741014) q[10];
ry(1.5693382993523015) q[11];
rz(1.8601251288091356) q[11];
ry(3.080716090921306) q[12];
rz(-0.034111876839770794) q[12];
ry(0.47257050119407096) q[13];
rz(-3.13657408893735) q[13];
ry(1.0465119003187286) q[14];
rz(-2.3697829699143003) q[14];
ry(1.5816222055655063) q[15];
rz(3.0806134996478036) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.555723112506823) q[0];
rz(-0.0047750776139568575) q[0];
ry(-1.575701721173762) q[1];
rz(2.975614751005979) q[1];
ry(3.138473094581321) q[2];
rz(1.610574560379468) q[2];
ry(-0.0025001599850904555) q[3];
rz(0.8802049192266042) q[3];
ry(-2.8735665061438675) q[4];
rz(0.0004275203786372032) q[4];
ry(7.539468754913514e-05) q[5];
rz(-0.14233282193609934) q[5];
ry(3.1378674373786004) q[6];
rz(0.659785012061822) q[6];
ry(-1.635814314766959) q[7];
rz(0.20053023054657082) q[7];
ry(-1.5662557249105262) q[8];
rz(1.5940562133800023) q[8];
ry(2.4480796916643226) q[9];
rz(0.04061207714570115) q[9];
ry(0.14628293418011773) q[10];
rz(-3.108625221355882) q[10];
ry(-2.1136104481267712) q[11];
rz(-3.063926878454316) q[11];
ry(-0.14796436396021895) q[12];
rz(-1.5777462828150977) q[12];
ry(1.1916645869769054) q[13];
rz(-1.5701302874021197) q[13];
ry(0.15081820119793743) q[14];
rz(-1.0809882361825833) q[14];
ry(1.6118658934713634) q[15];
rz(-0.23276022965889978) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.5721618784896043) q[0];
rz(0.04117538294256917) q[0];
ry(-3.1325024292905215) q[1];
rz(3.0141754848958535) q[1];
ry(-3.041778685236037) q[2];
rz(2.353278194285416) q[2];
ry(-1.0138507635132523) q[3];
rz(-1.6176427463348673) q[3];
ry(-1.607529165191651) q[4];
rz(-2.2639965549406633) q[4];
ry(3.141424870758342) q[5];
rz(1.1492953312789798) q[5];
ry(3.1392618116356084) q[6];
rz(-0.09239303970003032) q[6];
ry(0.004742341551265478) q[7];
rz(2.943766543387212) q[7];
ry(2.9477863880330735) q[8];
rz(0.009598411867728807) q[8];
ry(-3.1321557668463638) q[9];
rz(-3.0258489271805593) q[9];
ry(-3.136728411987164) q[10];
rz(0.12602192671287593) q[10];
ry(0.0131482968301313) q[11];
rz(1.5518239737361075) q[11];
ry(-1.5659758116430311) q[12];
rz(-2.1817735059344567) q[12];
ry(-1.5748251203939256) q[13];
rz(0.06789733118591142) q[13];
ry(2.9863512174317983) q[14];
rz(0.5063734728634754) q[14];
ry(-1.487990449973903) q[15];
rz(1.4288978096997416) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.5666045752651878) q[0];
rz(-0.0056248449969522425) q[0];
ry(1.669086201123189) q[1];
rz(1.8508643259194555) q[1];
ry(-0.0031292008306067487) q[2];
rz(-1.0090661973659163) q[2];
ry(3.133835224180325) q[3];
rz(-1.6074792780573173) q[3];
ry(2.9065382256826284) q[4];
rz(-2.2819665367487225) q[4];
ry(3.122685448268509) q[5];
rz(-0.11920110787789229) q[5];
ry(-1.6053388145761083) q[6];
rz(1.5688785153886475) q[6];
ry(-1.5925169923238007) q[7];
rz(-1.8297957772972495e-05) q[7];
ry(-1.9001149104463533) q[8];
rz(-0.772940458012992) q[8];
ry(-2.478270541747574) q[9];
rz(-1.4183186224927713) q[9];
ry(-1.660046141858854) q[10];
rz(-1.5241779025005178) q[10];
ry(1.676307336453533) q[11];
rz(-3.035047076767026) q[11];
ry(-2.6973054118383923) q[12];
rz(1.3665583065722888) q[12];
ry(1.532650797233194) q[13];
rz(1.7464238725185084) q[13];
ry(1.0850270529823192) q[14];
rz(-2.959881836985885) q[14];
ry(0.08885171334882039) q[15];
rz(-3.01334482714316) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.582722527943283) q[0];
rz(3.136953764029911) q[0];
ry(3.1159274051763446) q[1];
rz(2.942362195243335) q[1];
ry(0.007097613725509433) q[2];
rz(-2.065666943436547) q[2];
ry(-1.5789931270871125) q[3];
rz(-0.9641562414194705) q[3];
ry(1.5146313212944738) q[4];
rz(-2.4091868529520726) q[4];
ry(-1.524915350589349) q[5];
rz(2.2677962707427475) q[5];
ry(1.5692504702405676) q[6];
rz(-0.03397451507098962) q[6];
ry(-1.5682488256184401) q[7];
rz(-3.0819989119727964) q[7];
ry(-3.141563174369061) q[8];
rz(-0.6023162961442825) q[8];
ry(-3.1401085437146445) q[9];
rz(-2.5973604401477397) q[9];
ry(0.017618965582267693) q[10];
rz(-1.6129743014376903) q[10];
ry(-3.110888263288468) q[11];
rz(1.3690018813771483) q[11];
ry(3.136971886053821) q[12];
rz(1.4616229180557303) q[12];
ry(-1.5613670881890451) q[13];
rz(-0.0009956936608225564) q[13];
ry(-3.0643564618453865) q[14];
rz(1.6713757594382685) q[14];
ry(-1.6169304647640386) q[15];
rz(3.0354380598157538) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.5396024456572341) q[0];
rz(0.3607284828955088) q[0];
ry(-1.2399788657218571) q[1];
rz(0.919423979664697) q[1];
ry(0.019191046926287036) q[2];
rz(-2.116658291800971) q[2];
ry(3.1170939411832586) q[3];
rz(-0.11460262123618943) q[3];
ry(1.5711718640190289) q[4];
rz(3.1415283724398138) q[4];
ry(1.57081352934153) q[5];
rz(-0.0026784593945539115) q[5];
ry(3.0795520838336152) q[6];
rz(2.5482125914024616) q[6];
ry(-1.522682051140082) q[7];
rz(2.312124301881957) q[7];
ry(1.5614441589040675) q[8];
rz(2.2013267526174243) q[8];
ry(0.060571270803989385) q[9];
rz(-1.2302362787840941) q[9];
ry(1.6055617428331406) q[10];
rz(2.6763808982945365) q[10];
ry(-3.1207052086902296) q[11];
rz(2.172306645080803) q[11];
ry(3.139439938360248) q[12];
rz(1.30222574406361) q[12];
ry(-1.763468810903448) q[13];
rz(0.0021744763759437586) q[13];
ry(-1.5711006535446268) q[14];
rz(1.5776148015136473) q[14];
ry(1.5667119287154538) q[15];
rz(-1.5700320660211293) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(3.141381679061218) q[0];
rz(1.9170970423703066) q[0];
ry(3.135944946877408) q[1];
rz(2.553083358308546) q[1];
ry(-0.00021562889584824424) q[2];
rz(2.318766685245721) q[2];
ry(-3.1405661121174173) q[3];
rz(0.003656891272135699) q[3];
ry(1.571106412291222) q[4];
rz(1.6367164144380633) q[4];
ry(-1.5697766174036492) q[5];
rz(-3.0264420178851594) q[5];
ry(-0.001781475224751361) q[6];
rz(-0.25540007615653765) q[6];
ry(3.1368084480618856) q[7];
rz(0.874857235847477) q[7];
ry(-0.0011386550767972712) q[8];
rz(-0.8489920341861392) q[8];
ry(3.141484813636083) q[9];
rz(3.015119433773761) q[9];
ry(0.0005099566060327021) q[10];
rz(1.3511964788094146) q[10];
ry(-3.1258196190815606) q[11];
rz(1.0169190355726903) q[11];
ry(-3.079770051566136) q[12];
rz(2.473361438231874) q[12];
ry(-0.023830056058528767) q[13];
rz(1.4937450213097652) q[13];
ry(-3.089226545434) q[14];
rz(-1.532377972611719) q[14];
ry(-1.6294720494478432) q[15];
rz(-1.6548345787068506) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(0.3094929245507476) q[0];
rz(-0.5743108243820324) q[0];
ry(-1.9922173104881795) q[1];
rz(2.1968706549622743) q[1];
ry(0.1909739749097765) q[2];
rz(-2.354962918025032) q[2];
ry(0.13479512669562865) q[3];
rz(1.1594501987488517) q[3];
ry(2.4656127883271983) q[4];
rz(2.6231285934217214) q[4];
ry(1.8591347294483276) q[5];
rz(-0.4983987632898427) q[5];
ry(3.100836100221507) q[6];
rz(-2.957811543387911) q[6];
ry(2.270372326310892) q[7];
rz(1.9993750054509165) q[7];
ry(0.12124803394156825) q[8];
rz(-0.4309295429227351) q[8];
ry(-0.03677109049615942) q[9];
rz(2.5753073675538083) q[9];
ry(0.018047228087296734) q[10];
rz(1.7533143440208017) q[10];
ry(3.030158241970288) q[11];
rz(-1.220960615782566) q[11];
ry(0.05597629928841119) q[12];
rz(2.285887749762179) q[12];
ry(-1.6320258005704913) q[13];
rz(-2.2715921829169723) q[13];
ry(1.6179273890030375) q[14];
rz(2.937735411793881) q[14];
ry(1.610247776533206) q[15];
rz(0.6339294941076042) q[15];