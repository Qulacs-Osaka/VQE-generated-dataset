OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-2.6037083348320325) q[0];
rz(1.5382833683267014) q[0];
ry(3.0470888502616975) q[1];
rz(0.37733839654679774) q[1];
ry(-2.6398640218956646) q[2];
rz(-1.916070551913788) q[2];
ry(-0.5315906813665839) q[3];
rz(-2.3407320739298547) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.784991337630157) q[0];
rz(2.8889758903692506) q[0];
ry(1.985967886911907) q[1];
rz(0.2905220149455987) q[1];
ry(0.1734513617645561) q[2];
rz(-0.7132766043171914) q[2];
ry(1.5244671012617992) q[3];
rz(0.11219187810722583) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.156315218270924) q[0];
rz(1.1419860768075958) q[0];
ry(1.1648734351055678) q[1];
rz(0.36382146616241123) q[1];
ry(-1.1067889884770212) q[2];
rz(-1.4858120963637171) q[2];
ry(-0.28161947324826647) q[3];
rz(-0.53378502295749) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.0364399959270882) q[0];
rz(2.7385009162097993) q[0];
ry(-2.274976191426073) q[1];
rz(0.2288851428910066) q[1];
ry(2.3246501925510974) q[2];
rz(-0.6703287425448193) q[2];
ry(-0.2881416446531975) q[3];
rz(-1.5464684310098127) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.538982328405128) q[0];
rz(2.5864127758524162) q[0];
ry(2.6926973926530122) q[1];
rz(2.892989317290419) q[1];
ry(-2.465368378650769) q[2];
rz(0.10726054298198129) q[2];
ry(-3.0232030739117435) q[3];
rz(1.1134877985458969) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.1999199298347047) q[0];
rz(0.028667953963801732) q[0];
ry(3.0948266491031022) q[1];
rz(-0.08465494977751574) q[1];
ry(1.194829600434537) q[2];
rz(-2.140491680422916) q[2];
ry(-1.8670529644503604) q[3];
rz(2.022506074092397) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.7971158706393513) q[0];
rz(3.132771922951596) q[0];
ry(2.361209348060661) q[1];
rz(-2.222466624738366) q[1];
ry(-1.0248900309308622) q[2];
rz(0.851026586384406) q[2];
ry(-0.29864373458040117) q[3];
rz(1.1076871689644197) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.4147678457555628) q[0];
rz(-0.8582387222168164) q[0];
ry(0.3173561446780709) q[1];
rz(0.16302864814783646) q[1];
ry(3.0616957988513533) q[2];
rz(2.9209224967250087) q[2];
ry(-0.5072745148550797) q[3];
rz(-0.9724012547916346) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.7513412157544006) q[0];
rz(-2.7389970774337318) q[0];
ry(-2.632964571281157) q[1];
rz(-0.06689973705801353) q[1];
ry(0.8880128812067316) q[2];
rz(-2.0490585884134433) q[2];
ry(-2.3182714306639025) q[3];
rz(-0.4057055682595162) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(2.4720340943809704) q[0];
rz(1.3753457676336789) q[0];
ry(0.9397723525882977) q[1];
rz(-2.519123856679426) q[1];
ry(3.0979015030239836) q[2];
rz(1.524879997694093) q[2];
ry(2.775902560077472) q[3];
rz(-0.9213247771001297) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.3878483902850944) q[0];
rz(0.7536463910433683) q[0];
ry(-1.7970932741950136) q[1];
rz(1.482587872225479) q[1];
ry(-2.0704353434599074) q[2];
rz(0.6201943055860051) q[2];
ry(-2.625690704611092) q[3];
rz(-2.5583037663990753) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.141841091669533) q[0];
rz(0.5496800946811424) q[0];
ry(0.12252874627801079) q[1];
rz(-1.5435060500015372) q[1];
ry(2.691607554410308) q[2];
rz(-0.09748308862226042) q[2];
ry(-2.4822636479534093) q[3];
rz(0.8945315892844087) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.7895329987978092) q[0];
rz(-1.4885066755595342) q[0];
ry(-1.193749368209673) q[1];
rz(2.361663189654962) q[1];
ry(-0.6068426715812885) q[2];
rz(-2.297190692937069) q[2];
ry(0.232039172607424) q[3];
rz(-2.233717602523778) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.7468458808761086) q[0];
rz(-1.9554331763419324) q[0];
ry(2.5239263370680463) q[1];
rz(-2.569589654270459) q[1];
ry(0.0396918515139178) q[2];
rz(-2.4050847456193045) q[2];
ry(-2.9808443435450758) q[3];
rz(0.36499872804165207) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(3.0106289525591285) q[0];
rz(1.461119629228873) q[0];
ry(2.89809377964386) q[1];
rz(2.293585024148946) q[1];
ry(-0.1027554365479858) q[2];
rz(-2.450461804535387) q[2];
ry(1.24404243249698) q[3];
rz(0.6318838954908248) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.8422892657651495) q[0];
rz(-0.6728780278824518) q[0];
ry(2.2147036823537096) q[1];
rz(3.0216813135552893) q[1];
ry(-2.5315189792165302) q[2];
rz(-0.6966298614575582) q[2];
ry(2.6599214968072684) q[3];
rz(1.0348329719473872) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-0.5661924627136866) q[0];
rz(-2.7691438018616026) q[0];
ry(2.816211138782524) q[1];
rz(-1.1568625660850254) q[1];
ry(-2.8491971389259056) q[2];
rz(2.3410910779446965) q[2];
ry(1.8683701557668257) q[3];
rz(0.5448567638181921) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.6119358061691316) q[0];
rz(-2.2936411871153153) q[0];
ry(0.3565609097222647) q[1];
rz(-2.692523200490666) q[1];
ry(0.05287776859106304) q[2];
rz(1.125047689212889) q[2];
ry(0.5762117433163984) q[3];
rz(-1.9958548347648124) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.018539349347591) q[0];
rz(-1.1481878527486513) q[0];
ry(-1.6134654076704293) q[1];
rz(-0.7832694375252869) q[1];
ry(2.2266800984885147) q[2];
rz(2.5832306750371563) q[2];
ry(0.3055849801181039) q[3];
rz(0.48930304593016166) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.3731831061300896) q[0];
rz(1.7622007762534473) q[0];
ry(2.7375032686398257) q[1];
rz(-2.7726598236994766) q[1];
ry(-1.1852625696165007) q[2];
rz(-1.3248924717401418) q[2];
ry(1.3553156936935613) q[3];
rz(-0.991512368252666) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.3731596184633686) q[0];
rz(0.8331456311406233) q[0];
ry(-2.8336680250442527) q[1];
rz(0.46590039079679807) q[1];
ry(-0.5056228395395013) q[2];
rz(1.7529190661460263) q[2];
ry(-2.459434038328427) q[3];
rz(-0.19478485523018918) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(1.4373010885014355) q[0];
rz(-0.14093083049904553) q[0];
ry(2.855762412864344) q[1];
rz(2.5985833874281337) q[1];
ry(-0.17709165792585768) q[2];
rz(0.8875379693096743) q[2];
ry(1.7170128685349746) q[3];
rz(-1.1746413012657173) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-2.380715704074145) q[0];
rz(0.10613274015652417) q[0];
ry(-0.3322555695657572) q[1];
rz(1.1199096972320897) q[1];
ry(2.227959279176455) q[2];
rz(-0.5499032926920382) q[2];
ry(1.9780454889950863) q[3];
rz(-1.2520281233175123) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.2464581937876273) q[0];
rz(0.424356783176127) q[0];
ry(0.20054670907484784) q[1];
rz(-0.1046390885897077) q[1];
ry(2.6805117447092246) q[2];
rz(-2.205330483506679) q[2];
ry(-0.2590378871384891) q[3];
rz(-1.932381395587754) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-1.1187780920139314) q[0];
rz(2.2091194109182535) q[0];
ry(2.375828363802834) q[1];
rz(3.1288079775697804) q[1];
ry(2.373770243853007) q[2];
rz(0.2326534495654942) q[2];
ry(-2.661384520000117) q[3];
rz(3.0876764967214916) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.3962680053013372) q[0];
rz(-1.8481349560341362) q[0];
ry(-1.1784563512426578) q[1];
rz(0.7129369210280867) q[1];
ry(-2.277728003358157) q[2];
rz(-2.3852953865119892) q[2];
ry(-2.8215172152019385) q[3];
rz(0.6715421852494895) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(0.580085894885706) q[0];
rz(-2.637185682816321) q[0];
ry(-1.2332821493532373) q[1];
rz(2.7608185336686204) q[1];
ry(0.5452430283230951) q[2];
rz(1.8886573387606307) q[2];
ry(-2.6562342574406848) q[3];
rz(-2.6832711154130724) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[1],q[2];
ry(-3.1385837633314457) q[0];
rz(1.1447821568971301) q[0];
ry(1.747717798159045) q[1];
rz(-1.8394443745516809) q[1];
ry(1.4014032893405133) q[2];
rz(-2.6851293451228684) q[2];
ry(2.3946396891152237) q[3];
rz(-2.5207146373455496) q[3];