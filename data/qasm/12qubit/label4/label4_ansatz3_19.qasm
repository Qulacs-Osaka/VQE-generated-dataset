OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-2.061143950752573) q[0];
rz(0.8743516473089343) q[0];
ry(-1.9594822897206488) q[1];
rz(-2.0571158209808793) q[1];
ry(-1.868066815307991) q[2];
rz(-1.1191336992903957) q[2];
ry(3.141511852604629) q[3];
rz(0.8420740363599695) q[3];
ry(-0.0004974199928803952) q[4];
rz(-0.19866821112925415) q[4];
ry(1.5890892468957345) q[5];
rz(-1.9617600147423733) q[5];
ry(1.2926740775728158) q[6];
rz(-2.7764032562246967) q[6];
ry(-2.5625702736519926) q[7];
rz(-1.3248532579264154) q[7];
ry(1.073051455538366) q[8];
rz(1.9949469127645312) q[8];
ry(1.987084111235767) q[9];
rz(0.045147463746091354) q[9];
ry(2.3353116447227023) q[10];
rz(-2.289610648496134) q[10];
ry(-1.6771005748265417) q[11];
rz(0.20381196671243096) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-0.05522858489966165) q[0];
rz(-2.932376558451336) q[0];
ry(1.1741613044927197) q[1];
rz(0.41086170655164245) q[1];
ry(0.7048296653686972) q[2];
rz(-0.40007549227719785) q[2];
ry(-1.0862749816221025) q[3];
rz(-0.47582057449136855) q[3];
ry(2.4113537574778623) q[4];
rz(-2.328689894423511) q[4];
ry(0.057581488334057594) q[5];
rz(-3.0580462527827805) q[5];
ry(0.5861329436289164) q[6];
rz(-1.6882120106257972) q[6];
ry(-3.1325835922382526) q[7];
rz(1.7552722891161983) q[7];
ry(-2.56368611537131) q[8];
rz(1.6343879783999842) q[8];
ry(2.1263202192695654) q[9];
rz(1.4642177987877432) q[9];
ry(-0.6308396976338937) q[10];
rz(1.251666614353888) q[10];
ry(1.324750398905942) q[11];
rz(-2.0084488388483077) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-0.6032686771575481) q[0];
rz(-1.1133074314084332) q[0];
ry(0.3896620592829638) q[1];
rz(1.4879347784778343) q[1];
ry(3.1366022998362992) q[2];
rz(-2.065441972164654) q[2];
ry(3.1414371216802075) q[3];
rz(-2.7022159531332655) q[3];
ry(-3.1392532313745893) q[4];
rz(1.5260533484671681) q[4];
ry(3.1377971291666333) q[5];
rz(-2.8292715669047515) q[5];
ry(-2.4993847966139664) q[6];
rz(0.3102567334684476) q[6];
ry(-2.182394717556801) q[7];
rz(3.091467219570086) q[7];
ry(-0.8113836264518698) q[8];
rz(-2.2401324184803157) q[8];
ry(1.5231731043208832) q[9];
rz(-0.7717843641414506) q[9];
ry(-3.1411972308934772) q[10];
rz(1.6901903508238383) q[10];
ry(1.0918674959579748) q[11];
rz(-2.4508039010682494) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(2.7415618790359733) q[0];
rz(-0.6421968864117222) q[0];
ry(1.8394452838614876) q[1];
rz(1.674922743415122) q[1];
ry(-2.445099981531236) q[2];
rz(2.054094465932049) q[2];
ry(-0.12927168524550253) q[3];
rz(-0.09997283687480886) q[3];
ry(-2.4445157996603943) q[4];
rz(-2.726720435526327) q[4];
ry(-3.10343937353451) q[5];
rz(1.0627651490654522) q[5];
ry(1.3121758303850957) q[6];
rz(-0.886341234795605) q[6];
ry(-0.0006770980981469776) q[7];
rz(0.08155950828064998) q[7];
ry(2.3085349477913235) q[8];
rz(-2.584574413160934) q[8];
ry(1.968771785223502) q[9];
rz(-2.151966173312711) q[9];
ry(2.3055121434862826) q[10];
rz(-0.6016766072394873) q[10];
ry(-1.1655645100527838) q[11];
rz(1.9146069864611084) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(0.9270544983449582) q[0];
rz(-2.7739009010043736) q[0];
ry(-0.3236370354970679) q[1];
rz(2.4471599695354076) q[1];
ry(-4.920718571277405e-05) q[2];
rz(-0.393926690287877) q[2];
ry(-0.002808604171298461) q[3];
rz(-1.4860130728240621) q[3];
ry(-3.1391662009739743) q[4];
rz(-1.8023255587061988) q[4];
ry(-0.004341290694886091) q[5];
rz(2.677787770233302) q[5];
ry(1.1997250313533097) q[6];
rz(-0.0650003951793403) q[6];
ry(-2.250169016866513) q[7];
rz(1.7232631332072634) q[7];
ry(2.336280266145215) q[8];
rz(-1.9849626555071789) q[8];
ry(1.6908324135546593) q[9];
rz(1.4015780550539851) q[9];
ry(-1.6232036588448109) q[10];
rz(-1.8652126229586035) q[10];
ry(-1.2229353048242286) q[11];
rz(-1.793576946870792) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(2.644309741969693) q[0];
rz(-2.938088275326397) q[0];
ry(-1.3267234980828533) q[1];
rz(-2.70522018617528) q[1];
ry(-1.0809585888917437) q[2];
rz(2.511005042070956) q[2];
ry(1.9072083718482529) q[3];
rz(1.1961447398725877) q[3];
ry(-0.4995308013225186) q[4];
rz(-1.4883061685953898) q[4];
ry(1.597498307784698) q[5];
rz(-0.726462464367735) q[5];
ry(2.1023131499965357) q[6];
rz(1.7900315351098675) q[6];
ry(-3.135474383129807) q[7];
rz(-2.5278351490300746) q[7];
ry(-2.6293504424861243) q[8];
rz(1.9617341464905944) q[8];
ry(-0.42869103895948785) q[9];
rz(-0.4941407197163667) q[9];
ry(0.7521117500792885) q[10];
rz(-1.7252386339362393) q[10];
ry(-1.8945665657899022) q[11];
rz(-1.64538471387308) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-1.845914361361317) q[0];
rz(3.0844023218183323) q[0];
ry(-0.9197374825981317) q[1];
rz(-0.009305219749192515) q[1];
ry(-3.1404240480711056) q[2];
rz(2.7220403767371737) q[2];
ry(-3.14089230341041) q[3];
rz(2.728925671351166) q[3];
ry(-0.0006921893816205227) q[4];
rz(-1.7668879911793414) q[4];
ry(-3.1355087364803413) q[5];
rz(-0.3949593971247287) q[5];
ry(-0.9681466457439522) q[6];
rz(0.4336029481517842) q[6];
ry(-2.8033716130202984) q[7];
rz(-2.7663798028483964) q[7];
ry(-1.246780298313543) q[8];
rz(-0.5006146459819818) q[8];
ry(-0.8590616241553565) q[9];
rz(0.8747540564928249) q[9];
ry(-2.0360686583917653) q[10];
rz(-1.3719797009517392) q[10];
ry(-2.2035324735907382) q[11];
rz(-2.3916064388979463) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(2.088806298045943) q[0];
rz(1.1163703045999076) q[0];
ry(2.7571546959287847) q[1];
rz(1.40395297446426) q[1];
ry(1.990951541819732) q[2];
rz(1.5238018396362536) q[2];
ry(-0.27065406695026084) q[3];
rz(0.23124909984961187) q[3];
ry(-0.2575191224937958) q[4];
rz(2.7586915119462967) q[4];
ry(-0.054479317890803014) q[5];
rz(1.7456340295766442) q[5];
ry(1.8646576205195498) q[6];
rz(1.5667806090263374) q[6];
ry(0.0048043893411677755) q[7];
rz(-1.4132167263938815) q[7];
ry(2.309627763172339) q[8];
rz(0.8214106009440897) q[8];
ry(-2.822455941358954) q[9];
rz(-2.187739440662572) q[9];
ry(1.310225666104907) q[10];
rz(-2.2746130189217757) q[10];
ry(-0.24974680184777986) q[11];
rz(2.1886643613218615) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(0.8795831713199701) q[0];
rz(1.2446457628568819) q[0];
ry(0.7701385775966553) q[1];
rz(-1.1100918564963238) q[1];
ry(-1.8096713333099563) q[2];
rz(-0.009235314728910218) q[2];
ry(0.0007164538225552519) q[3];
rz(0.9946845695419987) q[3];
ry(-0.0014072028605067472) q[4];
rz(-1.3752244926433264) q[4];
ry(-0.07488682191478624) q[5];
rz(-2.619596758506867) q[5];
ry(1.490782467774121) q[6];
rz(-2.401661797921552) q[6];
ry(1.3027627383687914) q[7];
rz(-0.791675837460651) q[7];
ry(1.5987616138999958) q[8];
rz(-0.5199555651318493) q[8];
ry(-0.5094806269285136) q[9];
rz(-2.5395286142417604) q[9];
ry(-0.2079836164608852) q[10];
rz(0.27331079188858887) q[10];
ry(-1.5917083391336364) q[11];
rz(2.3588825289986586) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(2.1244074941415967) q[0];
rz(-0.0190639546336131) q[0];
ry(-2.6043145796627725) q[1];
rz(2.850016811593772) q[1];
ry(0.9497940944842865) q[2];
rz(2.1903903077379736) q[2];
ry(-0.04647851881164327) q[3];
rz(-1.2751831486241407) q[3];
ry(-3.1377930231779265) q[4];
rz(-0.45407874486658806) q[4];
ry(0.016176691693617722) q[5];
rz(2.8339293902167424) q[5];
ry(-1.2013736507590709) q[6];
rz(1.5422987069629466) q[6];
ry(3.133828188715298) q[7];
rz(-1.1235314046817857) q[7];
ry(1.6406403814313828) q[8];
rz(2.485982723790554) q[8];
ry(1.850854876462476) q[9];
rz(-0.8243209499185283) q[9];
ry(-1.9420228045867782) q[10];
rz(3.1190112323297554) q[10];
ry(-2.3717631990550747) q[11];
rz(-2.388614526687478) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(0.7597927177820285) q[0];
rz(-0.664773876868372) q[0];
ry(0.00037143028928721117) q[1];
rz(-0.4146894794885734) q[1];
ry(0.0022009646977760866) q[2];
rz(0.9613503308246644) q[2];
ry(0.0006276963943285807) q[3];
rz(-1.8004473832790318) q[3];
ry(-0.0006287018442989734) q[4];
rz(3.0989133067444747) q[4];
ry(3.1275647242320614) q[5];
rz(1.0427843840964666) q[5];
ry(0.30684942256975933) q[6];
rz(-0.47542561974154474) q[6];
ry(1.4594654412238706) q[7];
rz(2.608096030930973) q[7];
ry(3.1181548440517655) q[8];
rz(2.2464137550057908) q[8];
ry(1.851849934999267) q[9];
rz(-0.8794092300019463) q[9];
ry(-2.1576734116007343) q[10];
rz(-2.830929704094959) q[10];
ry(1.9338047108681935) q[11];
rz(-2.805727524266922) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-2.0364902332314907) q[0];
rz(0.544639101488677) q[0];
ry(-2.89993137614651) q[1];
rz(2.633784946497354) q[1];
ry(1.7031384426227498) q[2];
rz(2.1610541528135485) q[2];
ry(0.049059908119825) q[3];
rz(2.5773876138282623) q[3];
ry(1.5613764216434376) q[4];
rz(0.1642906398564179) q[4];
ry(3.1373059046599994) q[5];
rz(0.9532881071241687) q[5];
ry(0.1157016890493674) q[6];
rz(1.2062955707143734) q[6];
ry(3.1341855354457158) q[7];
rz(0.7926866373587582) q[7];
ry(-0.9843992580312353) q[8];
rz(1.8386792793713733) q[8];
ry(1.67345885878591) q[9];
rz(-0.6858873061161228) q[9];
ry(0.6412507544397216) q[10];
rz(-2.818999547725168) q[10];
ry(-1.825299029186956) q[11];
rz(-0.5087214256609565) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-0.009784045692020626) q[0];
rz(0.7195969463048745) q[0];
ry(0.0008127682350072926) q[1];
rz(-1.4689574430139007) q[1];
ry(1.2147129359263431e-05) q[2];
rz(-2.5044374949506265) q[2];
ry(-1.5672444893198991) q[3];
rz(-1.980757885833284) q[3];
ry(-0.006069146700111716) q[4];
rz(2.4418496569319883) q[4];
ry(-1.4545498999524393) q[5];
rz(-1.418084507515335) q[5];
ry(0.0017127757692669832) q[6];
rz(1.6279127457523765) q[6];
ry(-0.7876588884344224) q[7];
rz(1.4600644218110794) q[7];
ry(0.5286141630928416) q[8];
rz(0.7279975485840549) q[8];
ry(-2.0648905218222002) q[9];
rz(2.9985687497212616) q[9];
ry(1.3321485332873437) q[10];
rz(-1.6074783060142739) q[10];
ry(0.8234454760748955) q[11];
rz(-2.508710395907056) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-0.5966844696705778) q[0];
rz(1.4727669587195475) q[0];
ry(1.3272087922845586) q[1];
rz(0.8577443198782826) q[1];
ry(-0.0008056344456335295) q[2];
rz(-2.6171338361961967) q[2];
ry(0.0005520494033737222) q[3];
rz(-2.507437948561915) q[3];
ry(3.0400780984055875) q[4];
rz(-0.5428022374930903) q[4];
ry(2.781662119794509) q[5];
rz(-1.4276420253899316) q[5];
ry(1.8900008847556702) q[6];
rz(-0.48364112513059787) q[6];
ry(3.1409072335811876) q[7];
rz(-1.6360251295234807) q[7];
ry(1.4533611977520842) q[8];
rz(3.1135691758369157) q[8];
ry(-2.8838993273098734) q[9];
rz(0.7111481015572395) q[9];
ry(1.142081114744192) q[10];
rz(-1.044335167568132) q[10];
ry(-2.52874677730996) q[11];
rz(1.5673140895475663) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-0.14939068343418394) q[0];
rz(-1.552212169454637) q[0];
ry(2.8696230911562353) q[1];
rz(-2.4297362655236885) q[1];
ry(-3.1403198643702765) q[2];
rz(0.9827352544482535) q[2];
ry(-0.19226019699447633) q[3];
rz(-0.16113330999321035) q[3];
ry(3.1309423333265536) q[4];
rz(1.5668347110057117) q[4];
ry(1.3916347272953467) q[5];
rz(0.6480356328622405) q[5];
ry(-3.140147599786383) q[6];
rz(0.9382729537038328) q[6];
ry(-0.007852382750803244) q[7];
rz(0.7654769698001402) q[7];
ry(-0.5241256464315054) q[8];
rz(-2.6343344428566495) q[8];
ry(-1.7169553961663169) q[9];
rz(0.7630443015632914) q[9];
ry(-1.1641263752026338) q[10];
rz(0.5027445886914516) q[10];
ry(2.3776646328276714) q[11];
rz(1.834510018222998) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-0.0013235529364461482) q[0];
rz(-1.700977300720743) q[0];
ry(2.8955315939501496) q[1];
rz(0.03534172119926637) q[1];
ry(-3.140237492204816) q[2];
rz(0.7332738902960865) q[2];
ry(-3.1402507478933726) q[3];
rz(-3.0728697829226737) q[3];
ry(-0.6290176308000395) q[4];
rz(-1.3103758878062826) q[4];
ry(-1.46239307495725) q[5];
rz(0.89700804430103) q[5];
ry(1.1844641952820103) q[6];
rz(1.317452367760243) q[6];
ry(-3.1392660434356947) q[7];
rz(-1.4905477316601001) q[7];
ry(-0.8054143898181976) q[8];
rz(2.0696648671321176) q[8];
ry(-1.7703986184240088) q[9];
rz(-0.019880465561943297) q[9];
ry(-1.7603488595429386) q[10];
rz(1.7441743397301614) q[10];
ry(2.2058793762753783) q[11];
rz(-0.047246409769730276) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.272287173925756) q[0];
rz(-0.06991760415897906) q[0];
ry(-2.22148723658583) q[1];
rz(1.8076138885809223) q[1];
ry(2.046793740532971) q[2];
rz(-1.0067561768302662) q[2];
ry(1.5516371754980387) q[3];
rz(1.9790927139832935) q[3];
ry(-3.1415115675482457) q[4];
rz(0.7670905792999622) q[4];
ry(-0.36672123357385633) q[5];
rz(-0.9508739511585439) q[5];
ry(3.1336435420577957) q[6];
rz(-3.0837217614441674) q[6];
ry(-0.0008736948425149262) q[7];
rz(1.149852665807554) q[7];
ry(-1.2481010264334849) q[8];
rz(0.6257226621226888) q[8];
ry(1.2742016678804715) q[9];
rz(1.0046182606215766) q[9];
ry(-1.561864619941053) q[10];
rz(-0.32931317454056563) q[10];
ry(2.25219401433307) q[11];
rz(-2.0012651329741367) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-3.1292707341365236) q[0];
rz(-1.8786315479678708) q[0];
ry(2.89032823011007) q[1];
rz(2.630111495114401) q[1];
ry(-0.0024706721929117492) q[2];
rz(-2.3208836706862894) q[2];
ry(3.1404996216204264) q[3];
rz(-1.0230222533930018) q[3];
ry(0.00179711982464692) q[4];
rz(-0.5007679624223341) q[4];
ry(1.4030437020791848) q[5];
rz(0.17183224469896308) q[5];
ry(0.8474130120430212) q[6];
rz(-1.5077994065033167) q[6];
ry(-3.1405627621385017) q[7];
rz(-2.6070003521587752) q[7];
ry(0.07015389322127685) q[8];
rz(-2.8067708533865883) q[8];
ry(-1.3222221432899712) q[9];
rz(2.1607571716222442) q[9];
ry(-0.4051787115669109) q[10];
rz(2.828962957609882) q[10];
ry(-0.8789154354593229) q[11];
rz(-0.6903565253825059) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(-3.1357568312957445) q[0];
rz(-2.0265267982577515) q[0];
ry(-0.0036330467150755353) q[1];
rz(-0.5433127269009558) q[1];
ry(0.48085383131300635) q[2];
rz(1.0967108073151692) q[2];
ry(0.19739179735671097) q[3];
rz(3.007497310852641) q[3];
ry(-1.5669377347266016) q[4];
rz(-0.00030035161605837426) q[4];
ry(0.29684782372319685) q[5];
rz(0.2595440445921375) q[5];
ry(3.140676373089222) q[6];
rz(-1.1671178075844644) q[6];
ry(-2.512124053502626) q[7];
rz(2.0713376132216923) q[7];
ry(0.7382439688695657) q[8];
rz(2.207930261895818) q[8];
ry(0.382516800143783) q[9];
rz(1.8763483935131018) q[9];
ry(-1.1516704379987113) q[10];
rz(-1.6822213492449238) q[10];
ry(-1.1146394480711193) q[11];
rz(-2.6282653780171743) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(0.8655805648746134) q[0];
rz(-1.3434071780753687) q[0];
ry(-1.0915073958030033) q[1];
rz(-2.743948985367164) q[1];
ry(2.6979131803616947) q[2];
rz(-2.1947887657533536) q[2];
ry(2.8339654899401503) q[3];
rz(0.28997116816919716) q[3];
ry(1.7763962615289006) q[4];
rz(1.5754017063334782) q[4];
ry(0.00017868144281507628) q[5];
rz(2.765865453667594) q[5];
ry(-0.005666559218101774) q[6];
rz(-0.5051946577423853) q[6];
ry(-3.140428118770763) q[7];
rz(2.851570001024848) q[7];
ry(1.944820949797462) q[8];
rz(-0.7510211191792902) q[8];
ry(-1.0856615659244382) q[9];
rz(-0.013192861367929461) q[9];
ry(2.661947391570498) q[10];
rz(3.098049358724583) q[10];
ry(-0.16902323828966181) q[11];
rz(-0.13700270706201556) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(3.1409495440293163) q[0];
rz(0.07433951169686677) q[0];
ry(0.0008419290705727889) q[1];
rz(-1.4072984096294316) q[1];
ry(-0.0002719458863635893) q[2];
rz(0.9379467637185477) q[2];
ry(-3.1392947614224274) q[3];
rz(0.29181793824923474) q[3];
ry(-1.2566306309072122) q[4];
rz(-1.5722939911607352) q[4];
ry(3.0689434929140686) q[5];
rz(-0.616412733167591) q[5];
ry(-1.3931550903609704) q[6];
rz(-1.6212787187723539) q[6];
ry(-0.7474707748024155) q[7];
rz(1.0567116292786467) q[7];
ry(-0.006693064145087747) q[8];
rz(0.476215292471576) q[8];
ry(-0.6230771866104718) q[9];
rz(0.9196186225329175) q[9];
ry(2.257736117985236) q[10];
rz(-0.5306912362652474) q[10];
ry(-0.02984927056056162) q[11];
rz(-0.7645440270163991) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(1.3928624913822931) q[0];
rz(-0.581960770038173) q[0];
ry(-2.3019305391772824) q[1];
rz(-2.1370250151583687) q[1];
ry(0.4404034529307151) q[2];
rz(-1.7915868017164085) q[2];
ry(2.833850465799696) q[3];
rz(-2.9700087707138327) q[3];
ry(0.006644433160362588) q[4];
rz(-0.0007633770582771374) q[4];
ry(0.0004736927573256722) q[5];
rz(-2.5504842306687787) q[5];
ry(3.1353455227721367) q[6];
rz(-2.969351405706295) q[6];
ry(-0.002275710219424139) q[7];
rz(1.9772821344394083) q[7];
ry(-0.12752398504170462) q[8];
rz(1.1150914627813786) q[8];
ry(-1.0366411712282773) q[9];
rz(2.810817448797602) q[9];
ry(-0.3403520646351552) q[10];
rz(2.824172166298165) q[10];
ry(-0.8674455234720808) q[11];
rz(1.9163100126815815) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
ry(3.140643309978828) q[0];
rz(1.162941071250363) q[0];
ry(-3.139678666586493) q[1];
rz(-1.4531572901166294) q[1];
ry(-3.1409627885236358) q[2];
rz(-3.064032411087251) q[2];
ry(3.1410748028554507) q[3];
rz(2.8938712104250097) q[3];
ry(1.5774947088855242) q[4];
rz(1.5226873298496617) q[4];
ry(-3.0743651097756017) q[5];
rz(2.656293423942682) q[5];
ry(-0.6201558853579598) q[6];
rz(1.6447389859960162) q[6];
ry(0.5031327335263811) q[7];
rz(0.7964217530513132) q[7];
ry(-0.005673473732259538) q[8];
rz(2.4509562237271747) q[8];
ry(3.141084626952461) q[9];
rz(-0.5285099727533087) q[9];
ry(1.1962610779276617) q[10];
rz(3.0604884494066695) q[10];
ry(3.1280846759163556) q[11];
rz(1.3023395018670658) q[11];