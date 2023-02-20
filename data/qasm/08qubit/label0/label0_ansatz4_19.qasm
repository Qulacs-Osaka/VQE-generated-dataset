OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-0.34646207205996493) q[0];
rz(1.5855864406069742) q[0];
ry(2.6556040281472266) q[1];
rz(1.2491572546949679) q[1];
ry(-1.5710713183591212) q[2];
rz(-0.11373427647896865) q[2];
ry(-1.5704616532675786) q[3];
rz(-2.668323621031832) q[3];
ry(1.52474399448659) q[4];
rz(1.5261421015366692) q[4];
ry(1.1057135305042216) q[5];
rz(-0.8444356828522488) q[5];
ry(-0.4084677855747243) q[6];
rz(1.2096808178058103) q[6];
ry(2.7129658170407684) q[7];
rz(-0.49698118263261204) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.5905363573196434) q[0];
rz(0.4971176364783535) q[0];
ry(0.6136935897346499) q[1];
rz(-0.627651196433227) q[1];
ry(3.1386995181400374) q[2];
rz(1.3968332812598607) q[2];
ry(-0.003696047591926735) q[3];
rz(1.038188170006262) q[3];
ry(-1.0614214412451046) q[4];
rz(1.993804592246119) q[4];
ry(-1.830735145463949) q[5];
rz(-0.647706806461117) q[5];
ry(1.3396282945376388) q[6];
rz(1.5875029848035176) q[6];
ry(1.2716177114971625) q[7];
rz(-0.36521399674792715) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.027089590743445413) q[0];
rz(-2.376582090869031) q[0];
ry(1.941973271484989) q[1];
rz(0.3293171681607605) q[1];
ry(2.2548765784606255) q[2];
rz(-0.8870784097085855) q[2];
ry(-0.886802672830652) q[3];
rz(2.5099810949481283) q[3];
ry(2.4858423902484987) q[4];
rz(-2.394539795902855) q[4];
ry(2.1787902299793633) q[5];
rz(-2.502427688983676) q[5];
ry(-2.728666821328781) q[6];
rz(-0.6692541589697205) q[6];
ry(2.994270388700916) q[7];
rz(-1.3452499260450068) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.2847721504544705) q[0];
rz(-3.0407875210001634) q[0];
ry(0.050412718142742285) q[1];
rz(0.5222189516861858) q[1];
ry(0.00046989254391126164) q[2];
rz(-0.2046217571601501) q[2];
ry(3.1414142454253535) q[3];
rz(0.36566040810232714) q[3];
ry(0.8331853570233667) q[4];
rz(1.3211061715047934) q[4];
ry(1.031286694151584) q[5];
rz(-0.9459516037347075) q[5];
ry(-2.9794725210319957) q[6];
rz(0.6723190982160947) q[6];
ry(-2.0972635376009467) q[7];
rz(1.207111469189626) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.4653143306210739) q[0];
rz(0.5751672146530797) q[0];
ry(2.923259308241889) q[1];
rz(0.3191794211710697) q[1];
ry(-0.04008986709468414) q[2];
rz(2.3202997893416883) q[2];
ry(-0.03978720344038023) q[3];
rz(-1.6470096194489665) q[3];
ry(2.863896300896534) q[4];
rz(2.269106770231379) q[4];
ry(0.36761825178949703) q[5];
rz(-2.352442161510405) q[5];
ry(2.917095013343083) q[6];
rz(-1.0205034769870167) q[6];
ry(-2.8575102058585733) q[7];
rz(-1.6662510706012534) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.8555235247767756) q[0];
rz(-0.12839426922540742) q[0];
ry(-2.8718888445818993) q[1];
rz(1.6209391778672133) q[1];
ry(-1.9504494073174783e-06) q[2];
rz(-2.837070939811164) q[2];
ry(0.0003024690687276652) q[3];
rz(2.183859727523657) q[3];
ry(3.034343664757355) q[4];
rz(-0.565403994344327) q[4];
ry(0.08034146854836793) q[5];
rz(-2.7516441984627367) q[5];
ry(3.1253928371380617) q[6];
rz(-1.922486060900188) q[6];
ry(-1.609150005742745) q[7];
rz(1.9902779446046344) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.6151218585089557) q[0];
rz(-1.507433189542379) q[0];
ry(-0.033327259784964404) q[1];
rz(0.4455881168147988) q[1];
ry(1.5850815488458545) q[2];
rz(0.6355111692406252) q[2];
ry(-1.5671673444665757) q[3];
rz(0.589911925192872) q[3];
ry(-2.152856539910422) q[4];
rz(0.17173733290834292) q[4];
ry(-2.285009758577247) q[5];
rz(0.2783402247181055) q[5];
ry(-0.18207536561654974) q[6];
rz(-0.6788291328437304) q[6];
ry(0.8995171136794304) q[7];
rz(-0.026999392816273476) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.701569689898544) q[0];
rz(-2.640574597720218) q[0];
ry(1.3436225262485728) q[1];
rz(2.6788618047470374) q[1];
ry(-3.1411242885905795) q[2];
rz(2.9541825394542847) q[2];
ry(3.1405211186338473) q[3];
rz(3.12504585744349) q[3];
ry(-3.108886255267206) q[4];
rz(-2.1321125827328) q[4];
ry(2.3947513126317794) q[5];
rz(-3.1034902061858705) q[5];
ry(-0.5861095642765752) q[6];
rz(2.0425572424586456) q[6];
ry(0.710162536183045) q[7];
rz(-1.1094557622426615) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.6188050705594015) q[0];
rz(-2.15537261517305) q[0];
ry(2.1399630325339807) q[1];
rz(-1.4879345370287163) q[1];
ry(3.1397118328019165) q[2];
rz(0.31772973426864665) q[2];
ry(-0.001619349523754643) q[3];
rz(2.7478669228490826) q[3];
ry(2.168008465408005) q[4];
rz(1.8678693115210905) q[4];
ry(1.5211593719079355) q[5];
rz(-0.7552603250078197) q[5];
ry(3.0041638237677284) q[6];
rz(-1.6437290765898878) q[6];
ry(-0.7335893146740339) q[7];
rz(0.601921690986468) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.655465429211044) q[0];
rz(2.043371053765176) q[0];
ry(-0.20356493273666304) q[1];
rz(-2.1095402274813355) q[1];
ry(0.0007933385264106796) q[2];
rz(0.6967290896985397) q[2];
ry(0.0014122289962332734) q[3];
rz(-2.7468654354990196) q[3];
ry(1.258747963668206) q[4];
rz(0.47926595312313763) q[4];
ry(-2.4775456521705763) q[5];
rz(-0.2231163678100412) q[5];
ry(-2.900274308532494) q[6];
rz(3.101552622288956) q[6];
ry(-1.3547127002845907) q[7];
rz(-1.2044489255748125) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.7875404108183578) q[0];
rz(-2.025320705502273) q[0];
ry(-0.9859896609936429) q[1];
rz(-2.301103852096869) q[1];
ry(3.1413708998933365) q[2];
rz(-0.6118085033069178) q[2];
ry(-8.911066879664071e-05) q[3];
rz(-1.0009357289671774) q[3];
ry(0.7907644083820038) q[4];
rz(0.07769718866674505) q[4];
ry(-1.2672361292974237) q[5];
rz(2.878941187868828) q[5];
ry(-0.7673828100956831) q[6];
rz(-2.751650192551765) q[6];
ry(2.6261050888779023) q[7];
rz(-0.4255488440559949) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.9187770968965001) q[0];
rz(0.3657108906913491) q[0];
ry(-3.067270318053415) q[1];
rz(2.457962859878486) q[1];
ry(-0.0013991016402533358) q[2];
rz(2.410749770051249) q[2];
ry(0.0021545287676882907) q[3];
rz(1.5681815186390458) q[3];
ry(2.789837350328071) q[4];
rz(-3.0086070493767143) q[4];
ry(-1.027395114378606) q[5];
rz(-2.8525000336598056) q[5];
ry(-0.35885731508812135) q[6];
rz(1.804002340243045) q[6];
ry(1.348056735260319) q[7];
rz(-1.4859998546207667) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.8398034753327464) q[0];
rz(3.017810734342553) q[0];
ry(-2.6873720375108623) q[1];
rz(2.452172683253892) q[1];
ry(-1.5705314375705548) q[2];
rz(-1.5466127494961688) q[2];
ry(-1.570732348299508) q[3];
rz(-1.65987960052516) q[3];
ry(0.08068499266966223) q[4];
rz(-0.20504495980516138) q[4];
ry(1.166081901315457) q[5];
rz(0.35102041029679754) q[5];
ry(2.8014762974452965) q[6];
rz(2.5665046503223863) q[6];
ry(-2.3651433716024597) q[7];
rz(-2.4828350535763484) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(0.850376479938771) q[0];
rz(-3.0190137710998632) q[0];
ry(2.253200928422735) q[1];
rz(2.0870299746229777) q[1];
ry(-0.0006696457680329804) q[2];
rz(3.1175613390934607) q[2];
ry(-3.1413317910375103) q[3];
rz(-0.0881053429304517) q[3];
ry(1.9391045359222465) q[4];
rz(0.8785312538589684) q[4];
ry(-1.8589804064861868) q[5];
rz(-0.06974401569903041) q[5];
ry(0.9873226096664924) q[6];
rz(-2.961462670222605) q[6];
ry(2.2493251353644967) q[7];
rz(-2.809875436981996) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-2.378454855610919) q[0];
rz(-2.184928626086016) q[0];
ry(-1.5650900844266777) q[1];
rz(-1.5429945979124113) q[1];
ry(-1.5874560697841416) q[2];
rz(-2.024216048516738) q[2];
ry(-1.5875485668450076) q[3];
rz(-1.572128565323303) q[3];
ry(0.16708177153702883) q[4];
rz(2.8637555135379853) q[4];
ry(1.0758103494655968) q[5];
rz(2.2601446409299975) q[5];
ry(-2.5824277464561147) q[6];
rz(-1.864237126388911) q[6];
ry(0.6690813448027759) q[7];
rz(0.9845787379667473) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(2.160226965102644) q[0];
rz(-2.0130825124471556) q[0];
ry(-0.4195369602661135) q[1];
rz(-2.278790559440537) q[1];
ry(-3.1373462881829224) q[2];
rz(2.66481628580749) q[2];
ry(1.022249369336195) q[3];
rz(0.32628271825561056) q[3];
ry(0.3569991824882818) q[4];
rz(0.9657331548149459) q[4];
ry(-2.3209278090752155) q[5];
rz(2.5923612730416696) q[5];
ry(1.9486438681417197) q[6];
rz(0.9084871793391591) q[6];
ry(1.7671172973094373) q[7];
rz(1.848944004884894) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.07188046825027872) q[0];
rz(-0.5560805882289308) q[0];
ry(-2.6379384156301136) q[1];
rz(-0.40398164383240076) q[1];
ry(3.1332181052384724) q[2];
rz(3.101779151598331) q[2];
ry(0.002815993255834926) q[3];
rz(-0.3609857554216758) q[3];
ry(-2.8810360580629952) q[4];
rz(-1.132755753726291) q[4];
ry(-1.8351205590629736) q[5];
rz(0.659037858689663) q[5];
ry(2.7975579807050814) q[6];
rz(-2.4882879903033364) q[6];
ry(2.8888444593997957) q[7];
rz(-1.0673581609406853) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.04092014361761098) q[0];
rz(-2.2647884969239636) q[0];
ry(-0.5211423189013009) q[1];
rz(1.6818209210848956) q[1];
ry(-0.05857569714082001) q[2];
rz(-1.1260252801853412) q[2];
ry(0.05849599697249694) q[3];
rz(-1.6211595653576323) q[3];
ry(-1.314975227841371) q[4];
rz(-2.4508446935492247) q[4];
ry(-2.1544547903981206) q[5];
rz(1.2476264318229244) q[5];
ry(-1.6578871198689173) q[6];
rz(-0.603432677303778) q[6];
ry(-1.1713685862911394) q[7];
rz(1.389938290392827) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.4616190723501594) q[0];
rz(-3.1180619288908704) q[0];
ry(-1.5337112630150467) q[1];
rz(0.7259784462075877) q[1];
ry(2.7743692043273596) q[2];
rz(0.0681264814165841) q[2];
ry(0.4007203983406084) q[3];
rz(-3.089819135906227) q[3];
ry(2.6119284490043038) q[4];
rz(1.8495946668353138) q[4];
ry(0.5806252831214463) q[5];
rz(-0.9075183619916692) q[5];
ry(2.910973065818054) q[6];
rz(2.2220662837567566) q[6];
ry(2.3460749207548486) q[7];
rz(-1.1715877283517964) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(1.5683722281619126) q[0];
rz(-2.5550943384055147) q[0];
ry(-1.5415783893021129) q[1];
rz(1.820891330709812) q[1];
ry(-1.570376574935312) q[2];
rz(0.7213415829951897) q[2];
ry(1.5701924481887093) q[3];
rz(-1.569792311859433) q[3];
ry(2.54065094658851) q[4];
rz(0.820212522949834) q[4];
ry(2.87177443782695) q[5];
rz(1.712607331708902) q[5];
ry(-0.578022700533813) q[6];
rz(2.4030869034829103) q[6];
ry(1.957367375258303) q[7];
rz(1.8976429456386603) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.017570027599837168) q[0];
rz(2.4320760349536337) q[0];
ry(-3.1305677485195984) q[1];
rz(-1.3758428476638824) q[1];
ry(0.0003145099375446238) q[2];
rz(2.419096348418156) q[2];
ry(1.2300689582937097) q[3];
rz(1.5714380376161239) q[3];
ry(3.0307005343439193) q[4];
rz(2.713774520822752) q[4];
ry(-0.6651316237284818) q[5];
rz(-0.07766277610251786) q[5];
ry(-1.4784864207470627) q[6];
rz(1.5757299950330177) q[6];
ry(-3.1168956768585927) q[7];
rz(1.8033107647105067) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-1.5764273446183463) q[0];
rz(1.576847660009108) q[0];
ry(1.5772408198661687) q[1];
rz(1.4668055490284777) q[1];
ry(-1.5708072912766413) q[2];
rz(0.5988232241573169) q[2];
ry(-1.570806603550751) q[3];
rz(-0.06439343065080291) q[3];
ry(3.138644970131167) q[4];
rz(-0.5536276313593916) q[4];
ry(3.1413425101134838) q[5];
rz(1.6745838377073248) q[5];
ry(-1.6877709092654332) q[6];
rz(-1.0716457878223795) q[6];
ry(0.017700450197890977) q[7];
rz(2.546810949979449) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
ry(-0.22240375222035338) q[0];
rz(-1.5652569913409309) q[0];
ry(-0.04068132143229495) q[1];
rz(-1.5702186174524826) q[1];
ry(1.6350516777869633) q[2];
rz(-0.5058803102386371) q[2];
ry(0.6159243083371201) q[3];
rz(-1.414880273400109) q[3];
ry(-2.3150948026554894) q[4];
rz(2.3611110893905307) q[4];
ry(2.768024712917776) q[5];
rz(-1.4608138458403284) q[5];
ry(3.134181212417034) q[6];
rz(-2.6729860649332875) q[6];
ry(1.584360117695255) q[7];
rz(-0.0339176858322876) q[7];