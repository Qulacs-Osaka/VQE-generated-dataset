OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-2.429182302464492) q[0];
rz(1.3910545475244556) q[0];
ry(-2.456806906071364) q[1];
rz(2.6249427300912966) q[1];
ry(3.059450309595559) q[2];
rz(1.2955398068989832) q[2];
ry(0.29681194785922127) q[3];
rz(-2.63924630264062) q[3];
ry(1.2985539362791485) q[4];
rz(-0.6155924507187206) q[4];
ry(2.6427483740804334) q[5];
rz(-1.545667638110114) q[5];
ry(1.2444608418760126) q[6];
rz(0.39171885202451023) q[6];
ry(-2.4754087759510717) q[7];
rz(2.984151115118841) q[7];
ry(-3.0983942446288144) q[8];
rz(-0.6095841579212397) q[8];
ry(1.5737253890963947) q[9];
rz(1.4839457463636068) q[9];
ry(0.6695136580879124) q[10];
rz(2.1183072559897598) q[10];
ry(-0.07812455644415929) q[11];
rz(2.427907637779639) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.14961235287222682) q[0];
rz(-1.2760927903761772) q[0];
ry(0.022831711725712918) q[1];
rz(-0.29901880724642815) q[1];
ry(-0.40881971356046937) q[2];
rz(1.2258616837047382) q[2];
ry(0.19482030958216948) q[3];
rz(-2.8840661852809264) q[3];
ry(-2.5003219939258203) q[4];
rz(-0.7405912827222045) q[4];
ry(-0.004274992598243196) q[5];
rz(-1.8741937374746036) q[5];
ry(2.9767306177145265) q[6];
rz(-0.2906151810795512) q[6];
ry(2.4279180402693865) q[7];
rz(-0.1792950530013703) q[7];
ry(3.1329137893093377) q[8];
rz(-2.8755617253960963) q[8];
ry(0.2646633373274545) q[9];
rz(2.7325840258319847) q[9];
ry(1.4539567842305023) q[10];
rz(3.080815086287387) q[10];
ry(-1.7904941907358474) q[11];
rz(-0.7105174783657331) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.6737037823577586) q[0];
rz(-2.972335270414344) q[0];
ry(-3.1236441370167127) q[1];
rz(2.0460388416813906) q[1];
ry(-0.6981467306407515) q[2];
rz(0.5619324786513396) q[2];
ry(-0.020982122085042576) q[3];
rz(1.578150916364259) q[3];
ry(-0.7386205630112811) q[4];
rz(-2.1645735926706164) q[4];
ry(2.4242451090817085) q[5];
rz(0.5482558356973797) q[5];
ry(-2.982336081091414) q[6];
rz(0.6564497724963364) q[6];
ry(-2.9109633498746645) q[7];
rz(0.15536825587398367) q[7];
ry(2.4606717913153955) q[8];
rz(-1.590026679967644) q[8];
ry(1.7803050493676844) q[9];
rz(-1.6533980012464076) q[9];
ry(1.6230141906458406) q[10];
rz(3.0263533543964067) q[10];
ry(-1.5216182306709936) q[11];
rz(3.1293544849023216) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.833745701199493) q[0];
rz(-1.872052942989443) q[0];
ry(-0.2934031563387096) q[1];
rz(0.7414589328796923) q[1];
ry(-3.0847880957863314) q[2];
rz(0.27999891315789055) q[2];
ry(-0.1843423956284207) q[3];
rz(0.22195270888243188) q[3];
ry(0.4881974558815108) q[4];
rz(0.4479179551862304) q[4];
ry(-2.849857111331375) q[5];
rz(-1.4860333342041345) q[5];
ry(-2.986405650809522) q[6];
rz(0.9408399885981167) q[6];
ry(2.9398244828521216) q[7];
rz(2.005827167746454) q[7];
ry(0.19844671008318857) q[8];
rz(0.45214898970411466) q[8];
ry(0.8378797449463661) q[9];
rz(2.9049471444213757) q[9];
ry(-1.1405578723468635) q[10];
rz(0.7690123480067123) q[10];
ry(-2.35781110056427) q[11];
rz(-1.3506116496917995) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(3.1381468314738505) q[0];
rz(-1.006895361726868) q[0];
ry(3.127142499335167) q[1];
rz(-1.0976655549730792) q[1];
ry(-2.603792141391802) q[2];
rz(-1.7659555559263955) q[2];
ry(-0.18299461103272693) q[3];
rz(1.6067579870389246) q[3];
ry(-0.2957356859534121) q[4];
rz(0.3933982207629318) q[4];
ry(0.9721665018242875) q[5];
rz(-1.0638644870522895) q[5];
ry(-0.2391715479318437) q[6];
rz(2.152471786571551) q[6];
ry(1.5869748791576541) q[7];
rz(-2.675614484595805) q[7];
ry(-2.9922121655220555) q[8];
rz(-1.9048657838818146) q[8];
ry(-2.903695902893779) q[9];
rz(-0.6101146813268333) q[9];
ry(2.4808736403105986) q[10];
rz(2.5163661291066357) q[10];
ry(-1.6357844615392585) q[11];
rz(-2.8384009976691322) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.9875520434852225) q[0];
rz(-0.11219148506730431) q[0];
ry(0.10438811851041323) q[1];
rz(2.2966429170797587) q[1];
ry(-2.4571007025293037) q[2];
rz(-2.138968982842698) q[2];
ry(-2.856143408654882) q[3];
rz(-2.894354098613938) q[3];
ry(-0.5412900439134534) q[4];
rz(1.8435391172641196) q[4];
ry(2.1101642827108513) q[5];
rz(0.1030591891908683) q[5];
ry(-0.12953646271102492) q[6];
rz(0.14085893041115374) q[6];
ry(0.5510217443701331) q[7];
rz(0.34580772508510876) q[7];
ry(0.22429233096246876) q[8];
rz(2.146797840126223) q[8];
ry(0.20471196923748192) q[9];
rz(0.8765644744459937) q[9];
ry(0.17234884303777823) q[10];
rz(0.6443516063133538) q[10];
ry(-1.8035738386799816) q[11];
rz(2.2411398325502585) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.8402082172310172) q[0];
rz(0.16010844362473176) q[0];
ry(-3.120141624139228) q[1];
rz(2.5952977370722263) q[1];
ry(-2.124127564527973) q[2];
rz(-2.300496282843412) q[2];
ry(-3.1203494304296613) q[3];
rz(1.213927428773878) q[3];
ry(2.3221036654647937) q[4];
rz(-0.4145718395280265) q[4];
ry(-1.8449300869886924) q[5];
rz(-2.623749859579365) q[5];
ry(-3.1162602820825414) q[6];
rz(2.0901329943434246) q[6];
ry(-0.10648191231059824) q[7];
rz(2.5171579322155364) q[7];
ry(-1.8615146388839552) q[8];
rz(-1.00036754020533) q[8];
ry(2.1137363834150333) q[9];
rz(-0.6985230293065533) q[9];
ry(2.586561067435521) q[10];
rz(-1.0166955969243103) q[10];
ry(-0.3742520838587495) q[11];
rz(2.1275379441387585) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.7130682711383267) q[0];
rz(1.836213834449194) q[0];
ry(0.03606618711930043) q[1];
rz(-2.727486186692207) q[1];
ry(2.5747419758797543) q[2];
rz(1.0364699094258043) q[2];
ry(-0.19062192510703413) q[3];
rz(2.1290479021028164) q[3];
ry(0.9071540751200499) q[4];
rz(0.5261978385099297) q[4];
ry(-0.6067797986347407) q[5];
rz(2.8273647289582375) q[5];
ry(0.3059198338948656) q[6];
rz(0.7314795964078092) q[6];
ry(1.9187247669293432) q[7];
rz(2.829128823618467) q[7];
ry(-0.17972222839408938) q[8];
rz(2.3943045200130704) q[8];
ry(1.6351602963048941) q[9];
rz(-2.937484291993861) q[9];
ry(-0.7553697851621977) q[10];
rz(2.920820992213208) q[10];
ry(1.2773525838671755) q[11];
rz(-2.410513759644548) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.2134442136218366) q[0];
rz(2.2290012594776796) q[0];
ry(0.03687897770655635) q[1];
rz(-0.0733647854709707) q[1];
ry(-2.176758165296121) q[2];
rz(0.2679899185242249) q[2];
ry(2.156627874399886) q[3];
rz(-0.2629355147986842) q[3];
ry(-0.7651387610992301) q[4];
rz(-0.47894713994330423) q[4];
ry(0.834696715160062) q[5];
rz(-1.3279676220471277) q[5];
ry(0.8043249770892156) q[6];
rz(-0.3129704948414106) q[6];
ry(0.9287531050610172) q[7];
rz(-1.8785941031084867) q[7];
ry(-2.8261968193167357) q[8];
rz(-0.8506464586251568) q[8];
ry(1.6855836495996812) q[9];
rz(1.7298587925339035) q[9];
ry(0.005217883377035548) q[10];
rz(-0.648100098701069) q[10];
ry(2.4647164338699468) q[11];
rz(-1.5263631886561209) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.7588284206393858) q[0];
rz(-0.143377656144402) q[0];
ry(3.0900448496423003) q[1];
rz(-0.8656115315594166) q[1];
ry(0.2695636210980359) q[2];
rz(-2.454576667755044) q[2];
ry(-0.07700158268775414) q[3];
rz(-1.641622622316833) q[3];
ry(-0.15901198634462535) q[4];
rz(0.3371889293863628) q[4];
ry(0.004876767132459768) q[5];
rz(-0.689213188072524) q[5];
ry(3.056426192552906) q[6];
rz(1.4663925342856556) q[6];
ry(0.007302266369703325) q[7];
rz(0.2819888300959383) q[7];
ry(3.0504293555534505) q[8];
rz(-3.0692678035322136) q[8];
ry(-3.1304452825762814) q[9];
rz(1.6129445423123272) q[9];
ry(-2.62284932140085) q[10];
rz(-0.7816498445350133) q[10];
ry(1.4583930065005983) q[11];
rz(-1.781623532764978) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.949711828407308) q[0];
rz(2.823139697147199) q[0];
ry(-0.08585466947464619) q[1];
rz(-1.8402989480706158) q[1];
ry(0.21277990479701786) q[2];
rz(2.549018676928325) q[2];
ry(-2.1737066190942755) q[3];
rz(-2.4812034549238158) q[3];
ry(0.9320519557284631) q[4];
rz(0.03921052981837203) q[4];
ry(0.28838188316326535) q[5];
rz(-1.242886542767405) q[5];
ry(-2.8844524711625787) q[6];
rz(-3.0608895892392494) q[6];
ry(1.476960072673239) q[7];
rz(1.49547015737386) q[7];
ry(1.1753918624418032) q[8];
rz(-2.679993264529063) q[8];
ry(-1.185514555424617) q[9];
rz(0.9510203952948508) q[9];
ry(2.8223440988337773) q[10];
rz(-1.2833740650735237) q[10];
ry(1.5697392582365746) q[11];
rz(2.0715424562286726) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.009798141546870198) q[0];
rz(-2.4140251696149093) q[0];
ry(-0.9222001808929673) q[1];
rz(2.6541087610171137) q[1];
ry(2.10302132598136) q[2];
rz(-2.2552110415726307) q[2];
ry(-1.3989949203165828) q[3];
rz(0.6443795971205233) q[3];
ry(-0.8203449050133083) q[4];
rz(2.0924172662295053) q[4];
ry(0.31541691311792425) q[5];
rz(-1.7593445557276244) q[5];
ry(0.07506564375357339) q[6];
rz(-1.7458421703916045) q[6];
ry(-0.0003922925540235667) q[7];
rz(-2.26529009897829) q[7];
ry(-3.004963578480806) q[8];
rz(-0.6466126337340379) q[8];
ry(-2.0507708419678483) q[9];
rz(2.5031310477580484) q[9];
ry(-3.111687147404178) q[10];
rz(-0.12047237421839227) q[10];
ry(-1.8843301500736978) q[11];
rz(1.5998942496923554) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.06575677610443477) q[0];
rz(-0.5308331242480886) q[0];
ry(3.0086099657676515) q[1];
rz(-2.5544422299544447) q[1];
ry(3.121468521825533) q[2];
rz(-2.3595819989034026) q[2];
ry(-0.0008125370799181298) q[3];
rz(-3.033026490977785) q[3];
ry(3.108088021147454) q[4];
rz(-0.7172491150299553) q[4];
ry(-3.0796497605966984) q[5];
rz(1.5099019759423955) q[5];
ry(-0.8326395747241779) q[6];
rz(2.0564400006517136) q[6];
ry(-0.7262802199890185) q[7];
rz(-2.770892202694402) q[7];
ry(2.9261652128148343) q[8];
rz(2.5117515902510066) q[8];
ry(2.1522386769324138) q[9];
rz(2.2278638483158186) q[9];
ry(0.09697344036543676) q[10];
rz(-1.9581775519176419) q[10];
ry(-2.3645891796699514) q[11];
rz(0.02782059916380497) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.005852136952474218) q[0];
rz(-2.3197156000367007) q[0];
ry(2.8805107103141094) q[1];
rz(1.8934443791136824) q[1];
ry(-1.1726158405334044) q[2];
rz(-0.4047241619646468) q[2];
ry(2.804340907720902) q[3];
rz(-2.5738811087044775) q[3];
ry(-3.006015648992101) q[4];
rz(1.3067603020683962) q[4];
ry(1.94673875865503) q[5];
rz(2.283546469383585) q[5];
ry(-2.9789244355081235) q[6];
rz(-1.7744502303375942) q[6];
ry(-0.015394354794906917) q[7];
rz(-0.18054492481229903) q[7];
ry(3.024327814129108) q[8];
rz(0.6513878762982935) q[8];
ry(-1.6852187030647927) q[9];
rz(0.0989646381222693) q[9];
ry(0.12958785791809377) q[10];
rz(-1.7599684521056387) q[10];
ry(1.8269773582583753) q[11];
rz(-0.09440990539968674) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.6520925105232571) q[0];
rz(-0.8809594817601596) q[0];
ry(-0.348566662393135) q[1];
rz(0.5388933019767845) q[1];
ry(3.020531285254745) q[2];
rz(0.33284070374200514) q[2];
ry(0.9688348145135555) q[3];
rz(-1.9530393725461486) q[3];
ry(3.0469734063298173) q[4];
rz(1.006442556121744) q[4];
ry(3.091025844306822) q[5];
rz(2.8604006164873033) q[5];
ry(-0.07814566236578893) q[6];
rz(2.4214880940595354) q[6];
ry(0.3929195354398365) q[7];
rz(-1.828081787451612) q[7];
ry(-0.893417030363483) q[8];
rz(-1.0137007978196189) q[8];
ry(-2.123841242803085) q[9];
rz(-0.7783652960417725) q[9];
ry(3.0928216224768006) q[10];
rz(0.01070728229466944) q[10];
ry(-2.43713802512709) q[11];
rz(3.021775367793757) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.96901858012455) q[0];
rz(2.0116391922372174) q[0];
ry(-0.09538390997388468) q[1];
rz(1.919533507508896) q[1];
ry(-0.6703915196858086) q[2];
rz(-1.2678941536772106) q[2];
ry(0.08685887829855948) q[3];
rz(0.37349934056410816) q[3];
ry(1.59232534158158) q[4];
rz(0.012928779167587869) q[4];
ry(2.931933533291241) q[5];
rz(-2.510969607646187) q[5];
ry(-3.0146656395988765) q[6];
rz(1.3377687188434848) q[6];
ry(-2.978403774421892) q[7];
rz(-0.7659692312973458) q[7];
ry(3.0999382679755825) q[8];
rz(-3.045760380669572) q[8];
ry(-1.6260260449585073) q[9];
rz(-2.166974096327648) q[9];
ry(-0.38898408741721546) q[10];
rz(0.3885154193383551) q[10];
ry(-1.3064837863412617) q[11];
rz(1.5516731494839504) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.2672584083806786) q[0];
rz(-2.5669596045048624) q[0];
ry(-1.2063521670413921) q[1];
rz(-0.28693111740557775) q[1];
ry(-3.0658684456304544) q[2];
rz(-1.5032233583664425) q[2];
ry(0.7336032888244458) q[3];
rz(-3.08180783717113) q[3];
ry(3.038310976831775) q[4];
rz(0.12461232220929565) q[4];
ry(-1.915835935268499) q[5];
rz(2.931110323781107) q[5];
ry(1.490643775245764) q[6];
rz(1.239184151744736) q[6];
ry(-0.3009164188928972) q[7];
rz(-1.589823035292213) q[7];
ry(-0.34702533355130216) q[8];
rz(2.1088285885492453) q[8];
ry(-0.42590878094874896) q[9];
rz(0.4067351079031989) q[9];
ry(1.129054508426715) q[10];
rz(1.0532010243259553) q[10];
ry(-0.8953427877243465) q[11];
rz(2.693448564463766) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.0959727056220787) q[0];
rz(1.0037489107121313) q[0];
ry(-2.3369197727682365) q[1];
rz(1.2148139300019267) q[1];
ry(0.7841085811936215) q[2];
rz(-2.835636665817311) q[2];
ry(0.3161729498333337) q[3];
rz(-3.137717850936795) q[3];
ry(2.93662225760233) q[4];
rz(0.09382909579118816) q[4];
ry(1.3910184658993332) q[5];
rz(1.1774445590384408) q[5];
ry(-1.811758449186029) q[6];
rz(-0.003187581497616776) q[6];
ry(0.043534693868608976) q[7];
rz(0.492432385424391) q[7];
ry(-3.057186786734456) q[8];
rz(-0.18076319723106946) q[8];
ry(-2.545326546757228) q[9];
rz(2.6006098657807546) q[9];
ry(3.1134936263474824) q[10];
rz(1.477745490427276) q[10];
ry(-1.107849521072259) q[11];
rz(-1.136981809022658) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.7201490545107303) q[0];
rz(-2.2101975957430664) q[0];
ry(3.0216912105431724) q[1];
rz(-2.8537773678876492) q[1];
ry(2.793164465067443) q[2];
rz(2.7669498303549327) q[2];
ry(2.0282406577907333) q[3];
rz(0.11195315023744447) q[3];
ry(-1.5601183688377784) q[4];
rz(3.122664614242414) q[4];
ry(3.1219280595220766) q[5];
rz(0.9361322981642586) q[5];
ry(0.03693901670859492) q[6];
rz(-0.14706406738293598) q[6];
ry(3.043439327812261) q[7];
rz(1.4136845471798276) q[7];
ry(0.4627068352565046) q[8];
rz(0.5883861094130578) q[8];
ry(0.9021728213212341) q[9];
rz(-2.782529284519852) q[9];
ry(1.2614319729062167) q[10];
rz(-0.9864289494880153) q[10];
ry(1.341439277975283) q[11];
rz(0.8183996776815663) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.0968833059905387) q[0];
rz(2.2279972437710653) q[0];
ry(-1.3461217595313981) q[1];
rz(2.6770435581400247) q[1];
ry(-0.6619391538137693) q[2];
rz(1.9400989664160808) q[2];
ry(2.9440711258835752) q[3];
rz(-1.3478951112567183) q[3];
ry(-0.11542321239298836) q[4];
rz(0.7331836154316518) q[4];
ry(1.6792830347199088) q[5];
rz(3.104748812121108) q[5];
ry(1.8221854997911138) q[6];
rz(-1.6576228171567458) q[6];
ry(-1.1032487744424095) q[7];
rz(-3.1012185093477025) q[7];
ry(-2.0010084357064657) q[8];
rz(-2.8767302758321236) q[8];
ry(2.9830818911651518) q[9];
rz(1.2942471407194116) q[9];
ry(0.005211697278480714) q[10];
rz(-2.13832134140268) q[10];
ry(1.912131216844535) q[11];
rz(-0.26239638928959536) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.2117319823173169) q[0];
rz(-0.6119028802643106) q[0];
ry(-1.3606548019761782) q[1];
rz(1.964203750542664) q[1];
ry(0.0006396150605293518) q[2];
rz(1.8395214715449597) q[2];
ry(-0.036647239282565185) q[3];
rz(-1.9109502412643022) q[3];
ry(-0.007685392337836383) q[4];
rz(-1.166789609671753) q[4];
ry(0.7811313782220911) q[5];
rz(-0.009791896622732388) q[5];
ry(3.1321117688819915) q[6];
rz(-0.44309565990412203) q[6];
ry(-0.07837151451322806) q[7];
rz(-0.37213208748615273) q[7];
ry(-3.1361939001757677) q[8];
rz(-2.8773225069610677) q[8];
ry(-0.006529958529559387) q[9];
rz(1.9019346823489536) q[9];
ry(1.5521253621753228) q[10];
rz(-1.0438909265565615) q[10];
ry(-0.9040861755032782) q[11];
rz(1.5655035943203994) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.8621093180666843) q[0];
rz(-1.5039460759818866) q[0];
ry(2.2348314999399643) q[1];
rz(0.8937624334536435) q[1];
ry(-0.8004773296996428) q[2];
rz(0.08031503054932823) q[2];
ry(1.3779482903842497) q[3];
rz(-1.8205650806126288) q[3];
ry(3.118461615194902) q[4];
rz(-1.9592110217568282) q[4];
ry(-1.3313527457545726) q[5];
rz(1.7833827440009244) q[5];
ry(-0.07944405563739591) q[6];
rz(2.5309740286155646) q[6];
ry(-1.1634167628544763) q[7];
rz(1.881511919217946) q[7];
ry(2.0013646254117163) q[8];
rz(1.9092063636270984) q[8];
ry(1.7924414178644046) q[9];
rz(1.2780229213597236) q[9];
ry(0.18179642447157018) q[10];
rz(2.5213322444015454) q[10];
ry(-1.3108280941438182) q[11];
rz(-1.2305816848783033) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.337535034009158) q[0];
rz(-0.8728539227505322) q[0];
ry(2.4833865916967275) q[1];
rz(3.1158606921258563) q[1];
ry(-1.5711172508954003) q[2];
rz(3.135975450325922) q[2];
ry(0.07439929806701312) q[3];
rz(1.7617380769110564) q[3];
ry(2.6047352371733212) q[4];
rz(-3.098670436280835) q[4];
ry(-1.53338945920028) q[5];
rz(2.141339091433264) q[5];
ry(-0.021117878273256008) q[6];
rz(-1.3215804390141381) q[6];
ry(-0.5689253182229503) q[7];
rz(-0.11693940722075524) q[7];
ry(-3.1274415546662655) q[8];
rz(2.34398896501082) q[8];
ry(-1.7580393772797291) q[9];
rz(-2.229331924431457) q[9];
ry(-2.2923141957885798) q[10];
rz(-2.983308127941893) q[10];
ry(-1.8885221790262356) q[11];
rz(-2.815108975643046) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.7948737012678797) q[0];
rz(-1.5861197491550114) q[0];
ry(-1.5695064573597861) q[1];
rz(1.5719773108168722) q[1];
ry(3.0924895395836094) q[2];
rz(1.5659117637657163) q[2];
ry(-0.00022435446872393072) q[3];
rz(-1.490263549134683) q[3];
ry(-3.090334600043321) q[4];
rz(3.1131162781219177) q[4];
ry(0.03378659224212033) q[5];
rz(0.6459786574660731) q[5];
ry(3.126871525281617) q[6];
rz(2.747621945213093) q[6];
ry(0.34407546088128704) q[7];
rz(-3.0923169641749846) q[7];
ry(3.0894684462237003) q[8];
rz(2.063157003054717) q[8];
ry(0.08435672476018308) q[9];
rz(-2.206139868156432) q[9];
ry(-0.10674254790770836) q[10];
rz(-1.748101245829237) q[10];
ry(-0.9378914163361518) q[11];
rz(0.6623015617644548) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.00018144895672822514) q[0];
rz(0.28236923517854545) q[0];
ry(1.5711870956328093) q[1];
rz(0.01888868511661368) q[1];
ry(1.5708161543495576) q[2];
rz(2.441018629315463) q[2];
ry(-1.6397045380284738) q[3];
rz(1.9735430347816436) q[3];
ry(2.1008604830954107) q[4];
rz(-1.0621492478946557) q[4];
ry(0.1858275721462038) q[5];
rz(0.28872705509424945) q[5];
ry(-1.5807203732783757) q[6];
rz(-2.194816276470183) q[6];
ry(1.3946391731470025) q[7];
rz(-0.7284272530751928) q[7];
ry(-1.6114416019066518) q[8];
rz(-0.1330680006938547) q[8];
ry(1.3323692702611636) q[9];
rz(-2.5712630134795584) q[9];
ry(-1.81671494307907) q[10];
rz(1.4113707084522096) q[10];
ry(-1.8615483190259772) q[11];
rz(2.7104099429749953) q[11];