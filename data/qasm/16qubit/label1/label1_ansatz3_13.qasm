OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-1.758778962645999) q[0];
rz(2.7629015593944835) q[0];
ry(2.205137663572848) q[1];
rz(2.655060048559218) q[1];
ry(3.089980884787192) q[2];
rz(0.7555788660670527) q[2];
ry(-0.5289897332905067) q[3];
rz(0.12887028967931696) q[3];
ry(0.45421196220688675) q[4];
rz(2.4153970693574025) q[4];
ry(1.3935429702771804) q[5];
rz(2.725190229860987) q[5];
ry(0.23646778267448496) q[6];
rz(-1.3191043309845876) q[6];
ry(-0.8375317623952813) q[7];
rz(-0.5397569613321624) q[7];
ry(0.12361626960296694) q[8];
rz(-1.9344470891098187) q[8];
ry(3.140766111710013) q[9];
rz(-1.2025508049803528) q[9];
ry(0.2261518644118814) q[10];
rz(-3.0567788792091015) q[10];
ry(-2.919646959432226) q[11];
rz(-2.1196118914470325) q[11];
ry(-2.393446360986189) q[12];
rz(1.2979241514935649) q[12];
ry(-1.5589275120078083) q[13];
rz(-1.5625944645276029) q[13];
ry(0.6203878439214074) q[14];
rz(-0.8942036875706085) q[14];
ry(0.5102743353153265) q[15];
rz(-0.31563824026234316) q[15];
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
ry(1.4366297007062698) q[0];
rz(2.6065935407870064) q[0];
ry(-3.1396942834665342) q[1];
rz(0.3558812420039352) q[1];
ry(-0.01216825788493608) q[2];
rz(1.6698999039626048) q[2];
ry(-3.1412418570041907) q[3];
rz(-3.018909571844098) q[3];
ry(0.29669029314515644) q[4];
rz(-0.644730349940949) q[4];
ry(3.1413744665150243) q[5];
rz(-2.3035458440488217) q[5];
ry(-2.97965531759706) q[6];
rz(2.7424948601363446) q[6];
ry(-1.8902917698303376) q[7];
rz(0.51028419222291) q[7];
ry(-0.9047952364859366) q[8];
rz(-3.066009520417357) q[8];
ry(3.141117908233384) q[9];
rz(2.7257947542947636) q[9];
ry(-2.899440871431718) q[10];
rz(-3.101237363018591) q[10];
ry(0.0011723352311641124) q[11];
rz(1.2069314299786527) q[11];
ry(1.56008258962866) q[12];
rz(-3.1257086773517693) q[12];
ry(-2.704755937342627) q[13];
rz(1.5756781749587967) q[13];
ry(2.545467032055513) q[14];
rz(-0.30474835435997794) q[14];
ry(1.5695259251425853) q[15];
rz(0.009904187430391313) q[15];
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
ry(3.088751787795485) q[0];
rz(1.1962356228196915) q[0];
ry(-3.072664970710555) q[1];
rz(2.70024731508838) q[1];
ry(-2.269032942039943) q[2];
rz(1.7454275891648956) q[2];
ry(2.609798206085291) q[3];
rz(2.8284209312091804) q[3];
ry(-1.5062839467822222) q[4];
rz(1.8026754241810163) q[4];
ry(-0.9113759540537164) q[5];
rz(-0.7179758480899125) q[5];
ry(1.5560715116623802) q[6];
rz(-0.740887736849803) q[6];
ry(1.5875427345444706) q[7];
rz(2.148849301473973) q[7];
ry(2.1424067196164716) q[8];
rz(-2.4626777889877003) q[8];
ry(-0.0005046929566461245) q[9];
rz(-0.4122166480555194) q[9];
ry(-3.140696023159914) q[10];
rz(0.8877927242698894) q[10];
ry(-2.9765668693357727) q[11];
rz(0.8533614433940666) q[11];
ry(2.6202537113245876) q[12];
rz(-3.123231932964056) q[12];
ry(1.5720530172498868) q[13];
rz(0.8250999226855917) q[13];
ry(1.8076191792038643) q[14];
rz(0.967849122752848) q[14];
ry(1.397624239537735) q[15];
rz(0.000238175108888683) q[15];
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
ry(-3.0203310453547614) q[0];
rz(-2.3397275894913547) q[0];
ry(-2.0839842898599867) q[1];
rz(2.606690251764398) q[1];
ry(2.3408408681725508) q[2];
rz(1.7191822806030477) q[2];
ry(-3.141467911037359) q[3];
rz(2.9769605679087072) q[3];
ry(-3.1410628873486344) q[4];
rz(2.639085143766839) q[4];
ry(-0.24790145500031222) q[5];
rz(0.25894493537609864) q[5];
ry(-3.10012081782829) q[6];
rz(-0.8660136146286798) q[6];
ry(-1.5108885384829263) q[7];
rz(0.17347243823270356) q[7];
ry(-0.08229238511359505) q[8];
rz(0.5176127311480361) q[8];
ry(3.1413079165906033) q[9];
rz(-0.31451930370430614) q[9];
ry(-3.1347421629919294) q[10];
rz(0.421801415497689) q[10];
ry(-0.14517694255499428) q[11];
rz(1.9033801764947071) q[11];
ry(-2.39247897101167) q[12];
rz(1.5776349451073912) q[12];
ry(0.010232758841435013) q[13];
rz(0.06148193428990035) q[13];
ry(-1.5932180078424123) q[14];
rz(0.3077068322458484) q[14];
ry(-1.8741834025350146) q[15];
rz(3.141290844387427) q[15];
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
ry(-0.9618456696144397) q[0];
rz(0.9274954885584769) q[0];
ry(-1.02009750322215) q[1];
rz(-2.35881539296774) q[1];
ry(0.1867096729602009) q[2];
rz(0.42096299440144325) q[2];
ry(0.0023838393212232134) q[3];
rz(-1.0644533320661915) q[3];
ry(-1.4391011603321289) q[4];
rz(0.6469863349788518) q[4];
ry(-1.4990739180058945) q[5];
rz(0.15503558700838216) q[5];
ry(-1.186312005301897) q[6];
rz(-0.13386786211080537) q[6];
ry(-1.66500199736016) q[7];
rz(-1.8165955598950134) q[7];
ry(-2.960861138299643) q[8];
rz(-0.4397752016744984) q[8];
ry(1.5711017635505682) q[9];
rz(2.1576888166296007) q[9];
ry(0.0010060976527982105) q[10];
rz(-0.48211042881817257) q[10];
ry(1.8517899680554621) q[11];
rz(-3.014563429449589) q[11];
ry(1.5709990132861158) q[12];
rz(0.6174230477776539) q[12];
ry(-1.581378306989244) q[13];
rz(0.029604608214287967) q[13];
ry(0.8389468574211467) q[14];
rz(0.5842016879117107) q[14];
ry(-1.5742848408654229) q[15];
rz(1.5444495599892092) q[15];
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
ry(-2.8171447200063513) q[0];
rz(-1.273682340871491) q[0];
ry(1.502516263659744) q[1];
rz(-1.71057326631027) q[1];
ry(-1.6121377109676152) q[2];
rz(3.1060644723425) q[2];
ry(-3.1413978084509715) q[3];
rz(2.1918384165024074) q[3];
ry(3.133222441442173) q[4];
rz(-0.9377754041151497) q[4];
ry(3.1171506472020654) q[5];
rz(0.23893520258492276) q[5];
ry(3.139766161157624) q[6];
rz(2.208004566069472) q[6];
ry(0.00046822280873760566) q[7];
rz(-2.062198208854693) q[7];
ry(-1.572021323984801) q[8];
rz(1.5905337702715387) q[8];
ry(-3.1399396194402738) q[9];
rz(2.156845939721166) q[9];
ry(-0.0005130416619083533) q[10];
rz(1.3379204067755535) q[10];
ry(3.1391740158133654) q[11];
rz(2.252409524821431) q[11];
ry(1.8608740872027651) q[12];
rz(1.8250286191664369) q[12];
ry(2.8782214675846984) q[13];
rz(1.5919726419325626) q[13];
ry(-0.0005469583592780936) q[14];
rz(-0.16420052857051276) q[14];
ry(-2.889322244814757) q[15];
rz(3.1181779141823203) q[15];
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
ry(-0.08492447685481608) q[0];
rz(-2.0935237161568336) q[0];
ry(1.7878877956344423) q[1];
rz(2.8726732922312364) q[1];
ry(-2.3623803774586567) q[2];
rz(-1.5936393364345793) q[2];
ry(1.5742602809969357) q[3];
rz(1.6057262089368507) q[3];
ry(-1.533701374807606) q[4];
rz(-1.2859845784239972) q[4];
ry(-2.1611948735918105) q[5];
rz(2.969286775008117) q[5];
ry(3.1407127424540247) q[6];
rz(0.608379006356447) q[6];
ry(2.781879181314039) q[7];
rz(-2.047984268001051) q[7];
ry(-1.516870424512394) q[8];
rz(-1.6968795437823676) q[8];
ry(1.4390741452063944) q[9];
rz(-1.286476477202965) q[9];
ry(2.4070529402130294) q[10];
rz(-1.683957473186223) q[10];
ry(2.3410332455454386) q[11];
rz(1.1729123019556975) q[11];
ry(0.001364719269957071) q[12];
rz(-0.37852005193582716) q[12];
ry(-1.5076737152463613) q[13];
rz(1.5926038965854605) q[13];
ry(-2.0478113990489764) q[14];
rz(2.801910111859276) q[14];
ry(1.567083916517441) q[15];
rz(1.800269005675272) q[15];
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
ry(-1.5558288917810448) q[0];
rz(-3.108791615550229) q[0];
ry(0.0007073197673106435) q[1];
rz(-0.24985181438425297) q[1];
ry(-1.5743639496945423) q[2];
rz(0.0013447223549949427) q[2];
ry(3.1403069222731803) q[3];
rz(2.621313038817799) q[3];
ry(-3.1415325751904235) q[4];
rz(2.7785026757050293) q[4];
ry(-3.141563307603858) q[5];
rz(1.3124690002462645) q[5];
ry(2.8992697037345794e-06) q[6];
rz(2.8371857130017815) q[6];
ry(-1.5732505544559556) q[7];
rz(-0.36972073341459777) q[7];
ry(0.05700982669652195) q[8];
rz(1.612743147752739) q[8];
ry(-0.00321513696214204) q[9];
rz(-0.2849996374176209) q[9];
ry(-3.1273277105001136) q[10];
rz(0.15886628645361345) q[10];
ry(-3.0605268893260544) q[11];
rz(0.16756096034002385) q[11];
ry(0.9114138468699187) q[12];
rz(-1.60982758931958) q[12];
ry(-1.5758140725710943) q[13];
rz(0.9710370008036594) q[13];
ry(-2.692549937217926) q[14];
rz(0.8563324752281667) q[14];
ry(-1.3953727139279515) q[15];
rz(-1.4670833844206106) q[15];
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
ry(-0.2929915356616253) q[0];
rz(-1.8643585909486902) q[0];
ry(0.09919720718591929) q[1];
rz(-2.7780475224639614) q[1];
ry(-2.2012180931698486) q[2];
rz(1.2657482130385684) q[2];
ry(1.406759040061904) q[3];
rz(2.8615262227586458) q[3];
ry(-1.5622637507163573) q[4];
rz(-1.5567445212638784) q[4];
ry(-1.570839617320841) q[5];
rz(-0.5949335783674625) q[5];
ry(1.8812789874398195) q[6];
rz(-1.9928456811466022) q[6];
ry(0.005492986296449609) q[7];
rz(-2.777257968824786) q[7];
ry(0.05509552513025507) q[8];
rz(-2.8885888641358832) q[8];
ry(3.0072557958452983) q[9];
rz(-1.8714659394859259) q[9];
ry(-2.0276334799791176) q[10];
rz(0.6138776286759109) q[10];
ry(-3.133747313360611) q[11];
rz(-2.2133210121625266) q[11];
ry(1.649118412552772) q[12];
rz(0.30592136455133845) q[12];
ry(3.14102580704207) q[13];
rz(-1.9130782542473304) q[13];
ry(-1.5905857210820216) q[14];
rz(-0.058908810326368646) q[14];
ry(-2.986418832916606) q[15];
rz(-1.323426042562861) q[15];
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
ry(-3.105906177244545) q[0];
rz(-2.7036719357864634) q[0];
ry(-3.1211798227345824) q[1];
rz(-0.7622853475051379) q[1];
ry(0.03309413151797269) q[2];
rz(1.8747329446143484) q[2];
ry(3.140877682722672) q[3];
rz(-2.4856944862947854) q[3];
ry(0.2642599149754608) q[4];
rz(0.026004569769868976) q[4];
ry(-0.004607315136208036) q[5];
rz(-0.9690747204999005) q[5];
ry(3.1402198891487725) q[6];
rz(-2.646841798731542) q[6];
ry(-1.3980147168022228) q[7];
rz(2.428682734727922) q[7];
ry(-3.107286193724077) q[8];
rz(-1.1280765449443484) q[8];
ry(0.008327484948307706) q[9];
rz(-1.2581764718925073) q[9];
ry(3.0000894038282464) q[10];
rz(0.011707171278813093) q[10];
ry(2.849512405242634) q[11];
rz(-0.3463396850881812) q[11];
ry(-0.19039046968739684) q[12];
rz(-0.06615030745419316) q[12];
ry(-3.14011080236355) q[13];
rz(-1.1397766428993656) q[13];
ry(1.7053351645887909) q[14];
rz(1.5181827759888566) q[14];
ry(-1.4755144543140508) q[15];
rz(-2.4792515359879674) q[15];
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
ry(-1.7892536329509454) q[0];
rz(1.715943080604435) q[0];
ry(-1.595664055036732) q[1];
rz(0.6892186492411154) q[1];
ry(1.6996120997221726) q[2];
rz(1.5602853751694699) q[2];
ry(-1.1810654576286288) q[3];
rz(2.005261669709176) q[3];
ry(0.09194187520513619) q[4];
rz(-2.357281448662811) q[4];
ry(1.5596856093650253) q[5];
rz(-0.5356718723226339) q[5];
ry(1.567812881241817) q[6];
rz(0.611097231934945) q[6];
ry(0.056972679731576825) q[7];
rz(2.286898705695345) q[7];
ry(1.5789650951173284) q[8];
rz(0.23050222445013582) q[8];
ry(-0.08943472487628112) q[9];
rz(-0.17779300718992364) q[9];
ry(-1.0324151347556443) q[10];
rz(-1.801790218327663) q[10];
ry(1.5708673347282514) q[11];
rz(3.1207957980546372) q[11];
ry(-1.0803129942386533) q[12];
rz(2.026976240264678) q[12];
ry(3.140633047173877) q[13];
rz(1.5020414785288532) q[13];
ry(2.8110391710697993) q[14];
rz(2.536326367930789) q[14];
ry(-1.5190408813149698) q[15];
rz(1.5401217740075968) q[15];
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
ry(-0.040045339500990464) q[0];
rz(-1.9304406632857134) q[0];
ry(-0.040346853306933106) q[1];
rz(0.9485988445941294) q[1];
ry(-3.007371585010445) q[2];
rz(1.5862326174221817) q[2];
ry(-2.4841306835866606) q[3];
rz(3.1410546905789514) q[3];
ry(0.0025224921273119755) q[4];
rz(-0.8119682403792681) q[4];
ry(-3.1412910144502617) q[5];
rz(1.0690730486658868) q[5];
ry(-0.0017033790137208626) q[6];
rz(0.919597109723674) q[6];
ry(1.5657404321324502) q[7];
rz(-1.0447959441120622) q[7];
ry(3.139265708191771) q[8];
rz(-2.375862913155487) q[8];
ry(0.6195317237737639) q[9];
rz(1.3207547152444667) q[9];
ry(0.016126662752123266) q[10];
rz(1.8933491944423098) q[10];
ry(-2.9107497969103435) q[11];
rz(1.3865750330115363) q[11];
ry(1.8587462357731583) q[12];
rz(-2.9805314330905524) q[12];
ry(1.5732956896007748) q[13];
rz(1.830373405126796) q[13];
ry(-1.5426869897388231) q[14];
rz(-1.5619630949929852) q[14];
ry(-2.399622934839839) q[15];
rz(-1.9868510741224885) q[15];
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
ry(-1.5846611875845398) q[0];
rz(0.7874562302870989) q[0];
ry(-0.7029750294420127) q[1];
rz(-0.3214931840773206) q[1];
ry(-2.314806618270936) q[2];
rz(1.5646163746163921) q[2];
ry(-2.62949115506695) q[3];
rz(-0.007999450755211619) q[3];
ry(-0.4083668532404907) q[4];
rz(-1.5323944081111707) q[4];
ry(-0.03715351836581624) q[5];
rz(3.0729603887933257) q[5];
ry(1.7046847918053432) q[6];
rz(-2.5346634384575677) q[6];
ry(0.025352124597432635) q[7];
rz(-2.9429602406540263) q[7];
ry(-0.01340028009655424) q[8];
rz(1.5679552201854294) q[8];
ry(-3.138787985127877) q[9];
rz(-1.7883732186129269) q[9];
ry(2.951296695110651) q[10];
rz(-2.7683810341677746) q[10];
ry(-0.0021780498578625185) q[11];
rz(0.1522974705360367) q[11];
ry(-1.521918400972016) q[12];
rz(3.002652424387344) q[12];
ry(1.572309254033211) q[13];
rz(1.5701462185925594) q[13];
ry(1.580484957065913) q[14];
rz(-1.2573279138473819) q[14];
ry(-2.992646174471642) q[15];
rz(1.6367979423347876) q[15];
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
ry(1.3489089409384487) q[0];
rz(-0.7330787506341382) q[0];
ry(-3.1378230433912724) q[1];
rz(2.071602908130858) q[1];
ry(-1.5744389850994986) q[2];
rz(-0.3760011924767571) q[2];
ry(3.079245051541005) q[3];
rz(-0.0067608281100364) q[3];
ry(-3.1284911185130224) q[4];
rz(3.067441320883193) q[4];
ry(-3.1223820937962077) q[5];
rz(0.10734594830176113) q[5];
ry(-0.002621977669488018) q[6];
rz(-0.7324178890854558) q[6];
ry(-0.12178994517227638) q[7];
rz(-0.9944207677983826) q[7];
ry(3.138950923801386) q[8];
rz(-2.6024246856010276) q[8];
ry(-2.5097249379127207) q[9];
rz(1.7687409388566477) q[9];
ry(2.3478112829658393) q[10];
rz(2.7130184843695346) q[10];
ry(3.1014524695488537) q[11];
rz(1.559327412367481) q[11];
ry(1.5734748388307078) q[12];
rz(1.5977338233183556) q[12];
ry(-1.5704454292393315) q[13];
rz(2.959261072232727) q[13];
ry(-1.5942146752003452) q[14];
rz(1.4625741763476237) q[14];
ry(1.5792965767195337) q[15];
rz(-1.568304871504773) q[15];
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
ry(-2.1927071060626826) q[0];
rz(-2.3806944635413827) q[0];
ry(-1.807157854241962) q[1];
rz(0.7080264652875128) q[1];
ry(1.560501066029114) q[2];
rz(-0.1649306604851981) q[2];
ry(-1.5685005826287501) q[3];
rz(-1.068335727036197) q[3];
ry(-1.5810203065054527) q[4];
rz(2.553979784247679) q[4];
ry(-0.011572974257697766) q[5];
rz(-1.6907035689067684) q[5];
ry(-3.0069497840769435) q[6];
rz(-0.7954629769533765) q[6];
ry(-0.05565907321127206) q[7];
rz(-2.46354348640248) q[7];
ry(-2.843092665286461) q[8];
rz(-1.5496138016676673) q[8];
ry(-0.16181249104939255) q[9];
rz(-2.9014134925888944) q[9];
ry(1.6558632907916107) q[10];
rz(3.1399317750855653) q[10];
ry(3.059861540065596) q[11];
rz(3.1259214161330866) q[11];
ry(-0.027497975446435948) q[12];
rz(-1.307385058346239) q[12];
ry(0.6880483556256249) q[13];
rz(-2.961148490791256) q[13];
ry(-1.8867864980399922) q[14];
rz(-1.4168315796903588) q[14];
ry(1.488713471439504) q[15];
rz(-1.5796681437368905) q[15];
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
ry(1.5724141965805707) q[0];
rz(1.5693854837288816) q[0];
ry(-2.7710517012013836) q[1];
rz(2.279955176828663) q[1];
ry(0.040644834272662855) q[2];
rz(-1.4036469010751116) q[2];
ry(-1.6121077954618743e-05) q[3];
rz(-1.6447571996259462) q[3];
ry(-0.0009750075698775619) q[4];
rz(-0.9800526649501001) q[4];
ry(0.025290256644330947) q[5];
rz(3.11802579891568) q[5];
ry(0.0008619433979912828) q[6];
rz(0.6701956813727348) q[6];
ry(-1.6749275555600665) q[7];
rz(-0.04569582169208665) q[7];
ry(0.00846755603977801) q[8];
rz(3.119202633862885) q[8];
ry(-0.03808023154993556) q[9];
rz(-1.814727066010286) q[9];
ry(-2.2379421159101005) q[10];
rz(-0.0031367784784876052) q[10];
ry(-0.027413729305963624) q[11];
rz(1.586143919450361) q[11];
ry(0.008058584923829493) q[12];
rz(2.8537298156073465) q[12];
ry(3.1400203648691534) q[13];
rz(-2.5095495549247384) q[13];
ry(-2.8505388830986997) q[14];
rz(-0.03824191557180967) q[14];
ry(1.5714633501995492) q[15];
rz(3.036909278248067) q[15];
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
ry(1.5650537933340616) q[0];
rz(0.27955216674660266) q[0];
ry(-3.1399426359212383) q[1];
rz(-1.223235282363203) q[1];
ry(-1.5603431648297656) q[2];
rz(-1.4195566111246993) q[2];
ry(-0.005851819796689917) q[3];
rz(1.2618416960428513) q[3];
ry(-1.693704259156135) q[4];
rz(1.8353594530296444) q[4];
ry(1.582706562025099) q[5];
rz(1.8580368700109275) q[5];
ry(1.55431217594181) q[6];
rz(-1.2661817819371797) q[6];
ry(3.0660985623904873) q[7];
rz(-0.8511432398961472) q[7];
ry(-1.5738058801033308) q[8];
rz(-0.6379147322592046) q[8];
ry(1.5778620096158045) q[9];
rz(-2.399829860157738) q[9];
ry(-1.482715027019431) q[10];
rz(-1.7549884607682955) q[10];
ry(-1.572885152251637) q[11];
rz(0.6670950565090376) q[11];
ry(-1.568124972313032) q[12];
rz(-2.368214727634685) q[12];
ry(-3.139766657803552) q[13];
rz(-0.3297863218630077) q[13];
ry(-0.0018792261951820473) q[14];
rz(-2.213940839282853) q[14];
ry(-0.0009747138395264088) q[15];
rz(-0.7856967405859661) q[15];