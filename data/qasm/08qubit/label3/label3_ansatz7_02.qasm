OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-0.38383856557326945) q[0];
ry(-0.43634194546366) q[1];
cx q[0],q[1];
ry(-0.5487174708118037) q[0];
ry(2.479289544703612) q[1];
cx q[0],q[1];
ry(1.6844345079714051) q[0];
ry(-3.041010131903525) q[2];
cx q[0],q[2];
ry(2.5332759367622324) q[0];
ry(-1.9504977044601581) q[2];
cx q[0],q[2];
ry(1.932815949490137) q[0];
ry(-3.044953365117068) q[3];
cx q[0],q[3];
ry(1.6938814814375214) q[0];
ry(2.8430641128070637) q[3];
cx q[0],q[3];
ry(-3.098136771070896) q[0];
ry(-2.486676511288871) q[4];
cx q[0],q[4];
ry(2.8062519067435843) q[0];
ry(0.9878079882981158) q[4];
cx q[0],q[4];
ry(0.7158190129325561) q[0];
ry(0.4588906590092865) q[5];
cx q[0],q[5];
ry(0.1307165781067683) q[0];
ry(2.486608812108376) q[5];
cx q[0],q[5];
ry(2.395106088184616) q[0];
ry(-1.8971559206083741) q[6];
cx q[0],q[6];
ry(-2.0826021695657158) q[0];
ry(0.8067599198204531) q[6];
cx q[0],q[6];
ry(-0.21111813307930652) q[0];
ry(1.4056479139212057) q[7];
cx q[0],q[7];
ry(-1.0840984485995162) q[0];
ry(1.6878990884910294) q[7];
cx q[0],q[7];
ry(1.084261700376154) q[1];
ry(0.9663306074451761) q[2];
cx q[1],q[2];
ry(-1.3399321325321172) q[1];
ry(-1.6776506311846142) q[2];
cx q[1],q[2];
ry(-2.9153444277077205) q[1];
ry(1.1177803253031167) q[3];
cx q[1],q[3];
ry(1.1228111479188563) q[1];
ry(1.9971491469626157) q[3];
cx q[1],q[3];
ry(-2.000508182297687) q[1];
ry(-3.117947385601017) q[4];
cx q[1],q[4];
ry(1.6335929617009373) q[1];
ry(0.7499752973807278) q[4];
cx q[1],q[4];
ry(-2.9913051359358973) q[1];
ry(-2.252914118321094) q[5];
cx q[1],q[5];
ry(2.5631185053597707) q[1];
ry(1.3163908831373803) q[5];
cx q[1],q[5];
ry(-2.3349914509821286) q[1];
ry(-1.0146766507363365) q[6];
cx q[1],q[6];
ry(-1.1570771827607222) q[1];
ry(-0.6598193821087844) q[6];
cx q[1],q[6];
ry(2.181895281247199) q[1];
ry(-0.1842995889839978) q[7];
cx q[1],q[7];
ry(-1.0563475468334156) q[1];
ry(-2.7113635068936133) q[7];
cx q[1],q[7];
ry(0.23292909083799646) q[2];
ry(-1.6484575712200202) q[3];
cx q[2],q[3];
ry(1.2680175460856513) q[2];
ry(-2.983136432072227) q[3];
cx q[2],q[3];
ry(2.105351248219518) q[2];
ry(2.284178317948011) q[4];
cx q[2],q[4];
ry(1.6018312162340056) q[2];
ry(-0.3611502946631516) q[4];
cx q[2],q[4];
ry(-1.0408032423217952) q[2];
ry(0.05641560076310019) q[5];
cx q[2],q[5];
ry(-2.966620437278298) q[2];
ry(0.6966994507458715) q[5];
cx q[2],q[5];
ry(0.42900195981781675) q[2];
ry(-0.13799094787503388) q[6];
cx q[2],q[6];
ry(-1.52999153286428) q[2];
ry(2.182138053084053) q[6];
cx q[2],q[6];
ry(1.4994847297866538) q[2];
ry(-1.6383604054640568) q[7];
cx q[2],q[7];
ry(-0.7731900237671222) q[2];
ry(-1.2063874435001796) q[7];
cx q[2],q[7];
ry(-0.8596718908976877) q[3];
ry(-3.0010180599312006) q[4];
cx q[3],q[4];
ry(-2.298474421358598) q[3];
ry(2.6360657958300036) q[4];
cx q[3],q[4];
ry(-2.50718303477108) q[3];
ry(3.0758321856586264) q[5];
cx q[3],q[5];
ry(2.8760604248641393) q[3];
ry(2.990171880879946) q[5];
cx q[3],q[5];
ry(1.811904305518592) q[3];
ry(-1.3438906214192405) q[6];
cx q[3],q[6];
ry(2.310384839634748) q[3];
ry(-3.1143679307379624) q[6];
cx q[3],q[6];
ry(-2.0751472061067946) q[3];
ry(-2.0497635252541464) q[7];
cx q[3],q[7];
ry(2.98832258210643) q[3];
ry(1.3278970087219524) q[7];
cx q[3],q[7];
ry(3.112363838224781) q[4];
ry(1.2236471467457015) q[5];
cx q[4],q[5];
ry(1.737169706307836) q[4];
ry(1.8515105439525958) q[5];
cx q[4],q[5];
ry(2.112787866523912) q[4];
ry(2.6014431232829507) q[6];
cx q[4],q[6];
ry(-0.33756971736973096) q[4];
ry(-2.2825948326182814) q[6];
cx q[4],q[6];
ry(-1.007546736235362) q[4];
ry(-1.156982804799964) q[7];
cx q[4],q[7];
ry(2.4316316278740593) q[4];
ry(1.1850719241915633) q[7];
cx q[4],q[7];
ry(1.0061435897205646) q[5];
ry(-0.8999396470833867) q[6];
cx q[5],q[6];
ry(-0.32316519797554794) q[5];
ry(3.085912035985209) q[6];
cx q[5],q[6];
ry(1.9525558064892499) q[5];
ry(-1.9592713800415371) q[7];
cx q[5],q[7];
ry(1.4999272519054168) q[5];
ry(-0.7064125458977115) q[7];
cx q[5],q[7];
ry(0.3865909151442587) q[6];
ry(2.299644546912435) q[7];
cx q[6],q[7];
ry(0.3569361273777633) q[6];
ry(-1.863575034672172) q[7];
cx q[6],q[7];
ry(3.0690071568984347) q[0];
ry(-1.6608844329933528) q[1];
cx q[0],q[1];
ry(-0.11735120285635059) q[0];
ry(0.84032713290801) q[1];
cx q[0],q[1];
ry(1.9141222859754663) q[0];
ry(-2.463957659761611) q[2];
cx q[0],q[2];
ry(2.79957519655539) q[0];
ry(0.9502437387157849) q[2];
cx q[0],q[2];
ry(2.003447569634079) q[0];
ry(2.9578968349759953) q[3];
cx q[0],q[3];
ry(2.049311747282456) q[0];
ry(-0.32567330687977625) q[3];
cx q[0],q[3];
ry(0.6101155449801693) q[0];
ry(-0.6008985379049205) q[4];
cx q[0],q[4];
ry(-2.5187894240594964) q[0];
ry(-0.7641480374241328) q[4];
cx q[0],q[4];
ry(-2.2152417869727388) q[0];
ry(2.431099801000796) q[5];
cx q[0],q[5];
ry(1.975511170079114) q[0];
ry(-0.4047821555696025) q[5];
cx q[0],q[5];
ry(-2.291526685086867) q[0];
ry(-1.3740329685071417) q[6];
cx q[0],q[6];
ry(1.4162932787378386) q[0];
ry(0.5879507931435537) q[6];
cx q[0],q[6];
ry(1.1087418420373538) q[0];
ry(2.2467151594735215) q[7];
cx q[0],q[7];
ry(1.7665281653190004) q[0];
ry(2.8372411549274372) q[7];
cx q[0],q[7];
ry(-2.1651715894236236) q[1];
ry(1.203087644745467) q[2];
cx q[1],q[2];
ry(0.6784465904092617) q[1];
ry(-0.95710811043367) q[2];
cx q[1],q[2];
ry(2.2558940233970874) q[1];
ry(1.5973445594857114) q[3];
cx q[1],q[3];
ry(-0.9955648264850216) q[1];
ry(-1.834261940611639) q[3];
cx q[1],q[3];
ry(-2.1747769809182813) q[1];
ry(-1.0867447024980539) q[4];
cx q[1],q[4];
ry(-0.844529191746223) q[1];
ry(1.8707958337047659) q[4];
cx q[1],q[4];
ry(-2.62804655287832) q[1];
ry(-2.1984831193784826) q[5];
cx q[1],q[5];
ry(0.5309308814733891) q[1];
ry(1.0677162039136154) q[5];
cx q[1],q[5];
ry(0.501274351726177) q[1];
ry(3.030460944592352) q[6];
cx q[1],q[6];
ry(2.5192874189905186) q[1];
ry(1.7526659596036032) q[6];
cx q[1],q[6];
ry(1.4680119635718967) q[1];
ry(0.7801188931123277) q[7];
cx q[1],q[7];
ry(1.4517438768876278) q[1];
ry(1.267369178891359) q[7];
cx q[1],q[7];
ry(0.10120475877413194) q[2];
ry(-1.1550881387669036) q[3];
cx q[2],q[3];
ry(-3.0052253091259113) q[2];
ry(2.498984751570482) q[3];
cx q[2],q[3];
ry(0.7582612108553181) q[2];
ry(2.7897085002345254) q[4];
cx q[2],q[4];
ry(0.8476342325898425) q[2];
ry(1.7861529884059768) q[4];
cx q[2],q[4];
ry(0.7989224540495794) q[2];
ry(1.4400570366692824) q[5];
cx q[2],q[5];
ry(-0.1596163462413447) q[2];
ry(-0.908936744132439) q[5];
cx q[2],q[5];
ry(0.8825155130411969) q[2];
ry(0.6460648089777619) q[6];
cx q[2],q[6];
ry(2.526776381523071) q[2];
ry(-2.166852740120455) q[6];
cx q[2],q[6];
ry(-2.0075073741874987) q[2];
ry(-2.383163607641369) q[7];
cx q[2],q[7];
ry(0.22494543220310703) q[2];
ry(0.6471021810439321) q[7];
cx q[2],q[7];
ry(-1.134383279659679) q[3];
ry(-1.0289797317566967) q[4];
cx q[3],q[4];
ry(1.5414119658550176) q[3];
ry(0.7388879740605248) q[4];
cx q[3],q[4];
ry(0.9497466029222584) q[3];
ry(-3.105552389819485) q[5];
cx q[3],q[5];
ry(2.26123773526663) q[3];
ry(1.666390058440179) q[5];
cx q[3],q[5];
ry(0.16355905370944124) q[3];
ry(-2.7629682006030594) q[6];
cx q[3],q[6];
ry(2.1709632667887417) q[3];
ry(0.345175063015667) q[6];
cx q[3],q[6];
ry(2.9361099949077065) q[3];
ry(-0.5261658947483762) q[7];
cx q[3],q[7];
ry(1.0530905658902459) q[3];
ry(-2.0928229502268096) q[7];
cx q[3],q[7];
ry(1.6486276694891204) q[4];
ry(0.3810693206815757) q[5];
cx q[4],q[5];
ry(2.565756474956514) q[4];
ry(0.8531255629067037) q[5];
cx q[4],q[5];
ry(-0.3788780815561763) q[4];
ry(-2.955845938410796) q[6];
cx q[4],q[6];
ry(-0.6140489966553826) q[4];
ry(0.9449030785720105) q[6];
cx q[4],q[6];
ry(-2.583853504375902) q[4];
ry(-1.9155165856095393) q[7];
cx q[4],q[7];
ry(0.8406417603719896) q[4];
ry(2.206704609393677) q[7];
cx q[4],q[7];
ry(-1.453749547031479) q[5];
ry(2.4576653803341317) q[6];
cx q[5],q[6];
ry(-1.5721991445772439) q[5];
ry(2.073727673525698) q[6];
cx q[5],q[6];
ry(-0.16477423042930184) q[5];
ry(-2.9862930015108256) q[7];
cx q[5],q[7];
ry(-0.8599275268750769) q[5];
ry(-1.565822035807976) q[7];
cx q[5],q[7];
ry(1.599348665073741) q[6];
ry(0.9886668605256155) q[7];
cx q[6],q[7];
ry(-2.5485202637975646) q[6];
ry(-1.758481529641257) q[7];
cx q[6],q[7];
ry(-2.658448971097525) q[0];
ry(-2.8676854888603573) q[1];
cx q[0],q[1];
ry(-1.7994884814370082) q[0];
ry(-2.834776868738445) q[1];
cx q[0],q[1];
ry(0.8777700745809833) q[0];
ry(1.2818734658733741) q[2];
cx q[0],q[2];
ry(-1.600860664128362) q[0];
ry(-1.0685150136741282) q[2];
cx q[0],q[2];
ry(2.317929943517972) q[0];
ry(2.513307970195438) q[3];
cx q[0],q[3];
ry(-1.7388600933695122) q[0];
ry(1.6891842818701415) q[3];
cx q[0],q[3];
ry(-2.126064907848515) q[0];
ry(2.4377371152123475) q[4];
cx q[0],q[4];
ry(-2.9867186211276384) q[0];
ry(0.002550473969878176) q[4];
cx q[0],q[4];
ry(2.9793535290757607) q[0];
ry(1.3775444459779864) q[5];
cx q[0],q[5];
ry(-2.883270447091239) q[0];
ry(-1.4609024787000573) q[5];
cx q[0],q[5];
ry(1.4925968802721683) q[0];
ry(0.4879331800533411) q[6];
cx q[0],q[6];
ry(2.483335884684167) q[0];
ry(-1.4098223877200784) q[6];
cx q[0],q[6];
ry(-1.149280705403948) q[0];
ry(2.0687051106906824) q[7];
cx q[0],q[7];
ry(1.980168441394333) q[0];
ry(-1.7284298205935036) q[7];
cx q[0],q[7];
ry(-0.44399444354191436) q[1];
ry(1.578531994813577) q[2];
cx q[1],q[2];
ry(1.1186178929327093) q[1];
ry(-0.5495992242189187) q[2];
cx q[1],q[2];
ry(0.9493764058934004) q[1];
ry(1.7392891176289789) q[3];
cx q[1],q[3];
ry(-2.096703105282728) q[1];
ry(2.850179154253121) q[3];
cx q[1],q[3];
ry(1.9080084834919167) q[1];
ry(-2.2382949283198847) q[4];
cx q[1],q[4];
ry(2.322257490199573) q[1];
ry(-2.8911409657016773) q[4];
cx q[1],q[4];
ry(-2.8370846322909795) q[1];
ry(-1.7164654964723676) q[5];
cx q[1],q[5];
ry(-2.8289661619590003) q[1];
ry(-0.3145703623243632) q[5];
cx q[1],q[5];
ry(0.353402402559575) q[1];
ry(0.328638392146277) q[6];
cx q[1],q[6];
ry(1.8058480402823316) q[1];
ry(1.8304339589631002) q[6];
cx q[1],q[6];
ry(-0.1659377892788676) q[1];
ry(2.5644354338197366) q[7];
cx q[1],q[7];
ry(2.0005455294573338) q[1];
ry(-0.8107065050925315) q[7];
cx q[1],q[7];
ry(-2.3524633371392962) q[2];
ry(2.451136465201477) q[3];
cx q[2],q[3];
ry(0.16678471434521125) q[2];
ry(0.3586267808780814) q[3];
cx q[2],q[3];
ry(-2.590462870649746) q[2];
ry(1.9016716676611738) q[4];
cx q[2],q[4];
ry(-1.8691251595486378) q[2];
ry(2.8067662660843404) q[4];
cx q[2],q[4];
ry(2.719298615536325) q[2];
ry(-2.4377168561011118) q[5];
cx q[2],q[5];
ry(1.5330229852495882) q[2];
ry(0.7960420340274564) q[5];
cx q[2],q[5];
ry(1.4171832505905502) q[2];
ry(0.13957101702709007) q[6];
cx q[2],q[6];
ry(1.2526042697240103) q[2];
ry(0.8927170820145909) q[6];
cx q[2],q[6];
ry(0.5718445157443321) q[2];
ry(-0.8559817819597174) q[7];
cx q[2],q[7];
ry(-3.0559777951477494) q[2];
ry(-2.1614695657359846) q[7];
cx q[2],q[7];
ry(0.7137001592545795) q[3];
ry(2.3177920636436116) q[4];
cx q[3],q[4];
ry(1.214468888136108) q[3];
ry(0.19857004601749928) q[4];
cx q[3],q[4];
ry(1.3990158980384568) q[3];
ry(-2.716309150898483) q[5];
cx q[3],q[5];
ry(-2.2492477718520494) q[3];
ry(-1.5638134030461404) q[5];
cx q[3],q[5];
ry(-2.27426763687256) q[3];
ry(-1.9267154375116244) q[6];
cx q[3],q[6];
ry(-2.6289442930084768) q[3];
ry(-2.8107423346041753) q[6];
cx q[3],q[6];
ry(-0.473376215152426) q[3];
ry(-2.861114190722732) q[7];
cx q[3],q[7];
ry(1.125398438655381) q[3];
ry(0.04580228909012262) q[7];
cx q[3],q[7];
ry(-1.7131850432192994) q[4];
ry(-1.1751688752306564) q[5];
cx q[4],q[5];
ry(0.6784030876528209) q[4];
ry(1.8715527264531788) q[5];
cx q[4],q[5];
ry(1.3570621864649415) q[4];
ry(2.526083647433661) q[6];
cx q[4],q[6];
ry(-1.9081452253012374) q[4];
ry(-0.24703481552717585) q[6];
cx q[4],q[6];
ry(0.32647500154778175) q[4];
ry(0.8704786514246061) q[7];
cx q[4],q[7];
ry(-1.0272323999723993) q[4];
ry(0.12947565748209655) q[7];
cx q[4],q[7];
ry(1.8540540927361648) q[5];
ry(0.12451636550901223) q[6];
cx q[5],q[6];
ry(-1.09509619194648) q[5];
ry(-0.5160768682827818) q[6];
cx q[5],q[6];
ry(0.975741635413646) q[5];
ry(-2.717909007502797) q[7];
cx q[5],q[7];
ry(-0.16918446436772427) q[5];
ry(1.262873417964717) q[7];
cx q[5],q[7];
ry(-0.33441625804037) q[6];
ry(2.625785678540371) q[7];
cx q[6],q[7];
ry(2.8447851013437093) q[6];
ry(0.7795697623978611) q[7];
cx q[6],q[7];
ry(2.442701515793813) q[0];
ry(-0.9407997805660253) q[1];
cx q[0],q[1];
ry(1.6962010044683489) q[0];
ry(-2.9610227842846353) q[1];
cx q[0],q[1];
ry(-1.4965198281339758) q[0];
ry(0.9877143367912617) q[2];
cx q[0],q[2];
ry(-1.826375227105859) q[0];
ry(-0.2566757707457894) q[2];
cx q[0],q[2];
ry(-1.7412813217751093) q[0];
ry(-0.06646594851855259) q[3];
cx q[0],q[3];
ry(-2.5152536456172303) q[0];
ry(1.9338709202010973) q[3];
cx q[0],q[3];
ry(2.3779369744406917) q[0];
ry(-1.7844208940968231) q[4];
cx q[0],q[4];
ry(1.8461639590027286) q[0];
ry(-1.3397096548358371) q[4];
cx q[0],q[4];
ry(1.3209086722260857) q[0];
ry(1.5438048349895173) q[5];
cx q[0],q[5];
ry(2.660970957646332) q[0];
ry(-2.612649193159837) q[5];
cx q[0],q[5];
ry(2.198380249624025) q[0];
ry(1.3444886423040077) q[6];
cx q[0],q[6];
ry(-2.933860916043848) q[0];
ry(-3.114253076959896) q[6];
cx q[0],q[6];
ry(-2.4998232931715907) q[0];
ry(-0.3560004060028816) q[7];
cx q[0],q[7];
ry(-0.17481510251195018) q[0];
ry(-2.641755277753021) q[7];
cx q[0],q[7];
ry(-0.2000699340341364) q[1];
ry(2.8433749818488145) q[2];
cx q[1],q[2];
ry(-0.019232154037171028) q[1];
ry(-1.7076985176503587) q[2];
cx q[1],q[2];
ry(-2.990530819219346) q[1];
ry(1.1387359580321075) q[3];
cx q[1],q[3];
ry(-2.4422157793541484) q[1];
ry(0.07156772932472677) q[3];
cx q[1],q[3];
ry(1.1701561846974846) q[1];
ry(2.889216624659407) q[4];
cx q[1],q[4];
ry(-0.9679766770758507) q[1];
ry(-2.6283226356641562) q[4];
cx q[1],q[4];
ry(1.0059987047484444) q[1];
ry(-1.9553919068493775) q[5];
cx q[1],q[5];
ry(-1.8431136544359732) q[1];
ry(1.7691253439856487) q[5];
cx q[1],q[5];
ry(0.4012850887297396) q[1];
ry(-0.4874465526266935) q[6];
cx q[1],q[6];
ry(-1.9831128418599542) q[1];
ry(1.3582341755374343) q[6];
cx q[1],q[6];
ry(-1.4313666123923405) q[1];
ry(0.588296414766627) q[7];
cx q[1],q[7];
ry(-1.7235352303010592) q[1];
ry(-2.36917462715825) q[7];
cx q[1],q[7];
ry(-1.454696294243) q[2];
ry(-0.8790656342159826) q[3];
cx q[2],q[3];
ry(-0.5045510760458966) q[2];
ry(-1.3209017573790496) q[3];
cx q[2],q[3];
ry(2.338130838412009) q[2];
ry(-3.069259391391628) q[4];
cx q[2],q[4];
ry(1.1559941240726674) q[2];
ry(-2.377628487024018) q[4];
cx q[2],q[4];
ry(-1.9740681471603168) q[2];
ry(2.1120083566244396) q[5];
cx q[2],q[5];
ry(-2.8208939219808005) q[2];
ry(0.13459138424793507) q[5];
cx q[2],q[5];
ry(1.5771183497528032) q[2];
ry(-0.07125958547551738) q[6];
cx q[2],q[6];
ry(1.8609473587302876) q[2];
ry(1.331803859740072) q[6];
cx q[2],q[6];
ry(1.2219539701354059) q[2];
ry(-2.900499513763504) q[7];
cx q[2],q[7];
ry(-1.5467700692710373) q[2];
ry(2.736348738886141) q[7];
cx q[2],q[7];
ry(-2.757207432019315) q[3];
ry(-2.1497195967017664) q[4];
cx q[3],q[4];
ry(-2.7277423741442806) q[3];
ry(-1.6735264028362238) q[4];
cx q[3],q[4];
ry(-2.9139040762599637) q[3];
ry(1.2209828020661186) q[5];
cx q[3],q[5];
ry(-3.091073270819671) q[3];
ry(-2.017770179518174) q[5];
cx q[3],q[5];
ry(2.0547373145680297) q[3];
ry(2.86124035659424) q[6];
cx q[3],q[6];
ry(1.5719914149762273) q[3];
ry(0.5819001834090055) q[6];
cx q[3],q[6];
ry(-1.7060559718875945) q[3];
ry(-2.429654953238529) q[7];
cx q[3],q[7];
ry(1.591604837028013) q[3];
ry(-1.2119650648497473) q[7];
cx q[3],q[7];
ry(1.6120729202386312) q[4];
ry(2.9797482943719897) q[5];
cx q[4],q[5];
ry(0.24790227556839017) q[4];
ry(0.4895718551579815) q[5];
cx q[4],q[5];
ry(2.630605196160822) q[4];
ry(-1.466468192165691) q[6];
cx q[4],q[6];
ry(-2.0115881219755756) q[4];
ry(3.0659174410078784) q[6];
cx q[4],q[6];
ry(-2.0131076591191466) q[4];
ry(-2.4989446312652226) q[7];
cx q[4],q[7];
ry(-0.3418702981793116) q[4];
ry(2.5782229276606308) q[7];
cx q[4],q[7];
ry(2.1820372321500443) q[5];
ry(-2.3382848097317805) q[6];
cx q[5],q[6];
ry(0.35891165209798337) q[5];
ry(1.7790816646158427) q[6];
cx q[5],q[6];
ry(-0.2608967279265899) q[5];
ry(-0.06810772053061571) q[7];
cx q[5],q[7];
ry(1.790287789331467) q[5];
ry(0.29752867136045413) q[7];
cx q[5],q[7];
ry(1.7813745646068906) q[6];
ry(-1.0562081180871221) q[7];
cx q[6],q[7];
ry(-0.1865720412845649) q[6];
ry(1.1828422376055707) q[7];
cx q[6],q[7];
ry(2.420911426134643) q[0];
ry(1.9813829590050016) q[1];
cx q[0],q[1];
ry(-2.1267954898684933) q[0];
ry(2.323073268433316) q[1];
cx q[0],q[1];
ry(2.761008256536467) q[0];
ry(2.146811155854115) q[2];
cx q[0],q[2];
ry(-3.0470310868352226) q[0];
ry(1.257806618070517) q[2];
cx q[0],q[2];
ry(-0.4870885479397874) q[0];
ry(2.483081175757829) q[3];
cx q[0],q[3];
ry(-1.1501366145913248) q[0];
ry(0.37066145550180596) q[3];
cx q[0],q[3];
ry(1.72298965530572) q[0];
ry(2.295227954763346) q[4];
cx q[0],q[4];
ry(-0.3072342055452957) q[0];
ry(-1.692864481447013) q[4];
cx q[0],q[4];
ry(0.7973633401384388) q[0];
ry(1.314336328595078) q[5];
cx q[0],q[5];
ry(-0.20062437379628673) q[0];
ry(2.885685256337752) q[5];
cx q[0],q[5];
ry(0.08892840809930558) q[0];
ry(2.503720340538485) q[6];
cx q[0],q[6];
ry(1.613847202019012) q[0];
ry(0.37199300681725916) q[6];
cx q[0],q[6];
ry(-0.9270689322245959) q[0];
ry(-2.681679133111693) q[7];
cx q[0],q[7];
ry(0.8423991778524513) q[0];
ry(-2.625303025881802) q[7];
cx q[0],q[7];
ry(1.120604179055266) q[1];
ry(-3.0550031134426097) q[2];
cx q[1],q[2];
ry(0.7346715015516178) q[1];
ry(-1.7236744772604728) q[2];
cx q[1],q[2];
ry(1.7822788827613307) q[1];
ry(-0.5496771319057646) q[3];
cx q[1],q[3];
ry(-0.32021004232879857) q[1];
ry(-2.1763610130586772) q[3];
cx q[1],q[3];
ry(1.2762370365306186) q[1];
ry(0.6722935224900597) q[4];
cx q[1],q[4];
ry(2.050539997496309) q[1];
ry(-0.8988780762683823) q[4];
cx q[1],q[4];
ry(1.5499861266492276) q[1];
ry(2.518100034203776) q[5];
cx q[1],q[5];
ry(1.3874409616358214) q[1];
ry(2.06175334727957) q[5];
cx q[1],q[5];
ry(-0.6013800423965474) q[1];
ry(-1.6753432992586939) q[6];
cx q[1],q[6];
ry(0.5763511461568918) q[1];
ry(0.2590915465657182) q[6];
cx q[1],q[6];
ry(-3.01378336350482) q[1];
ry(3.0526758833118164) q[7];
cx q[1],q[7];
ry(2.651628403784069) q[1];
ry(0.7743569030263817) q[7];
cx q[1],q[7];
ry(-1.821209848063261) q[2];
ry(2.0145650213934383) q[3];
cx q[2],q[3];
ry(-1.3992272736436067) q[2];
ry(3.134380139990533) q[3];
cx q[2],q[3];
ry(-3.0319977372612477) q[2];
ry(1.9121199451264246) q[4];
cx q[2],q[4];
ry(2.627837497791766) q[2];
ry(-2.9033663097407927) q[4];
cx q[2],q[4];
ry(2.438953784815348) q[2];
ry(0.94052967621709) q[5];
cx q[2],q[5];
ry(0.08305959018009813) q[2];
ry(0.44272490008648896) q[5];
cx q[2],q[5];
ry(-1.4284111121728331) q[2];
ry(0.3747400554649947) q[6];
cx q[2],q[6];
ry(0.033406234036258) q[2];
ry(1.054045340895501) q[6];
cx q[2],q[6];
ry(3.0898440255575492) q[2];
ry(-0.1900931532863419) q[7];
cx q[2],q[7];
ry(-1.6235293065127605) q[2];
ry(1.8484954425669962) q[7];
cx q[2],q[7];
ry(-1.6015556527008084) q[3];
ry(0.8276806656088608) q[4];
cx q[3],q[4];
ry(-2.354022899237682) q[3];
ry(1.239281378558461) q[4];
cx q[3],q[4];
ry(2.327971100779916) q[3];
ry(2.7540430971304555) q[5];
cx q[3],q[5];
ry(-2.2923058087859465) q[3];
ry(0.16303924971585637) q[5];
cx q[3],q[5];
ry(1.9252459236346142) q[3];
ry(-2.356343870662075) q[6];
cx q[3],q[6];
ry(-3.054217990225371) q[3];
ry(-1.8500146463799105) q[6];
cx q[3],q[6];
ry(1.7459671741877987) q[3];
ry(-0.4397206130450589) q[7];
cx q[3],q[7];
ry(-1.1937815406201802) q[3];
ry(-3.0823593414405503) q[7];
cx q[3],q[7];
ry(2.9174957086994757) q[4];
ry(-2.8567009041449083) q[5];
cx q[4],q[5];
ry(2.113860142227498) q[4];
ry(2.8344340301922393) q[5];
cx q[4],q[5];
ry(2.1161056121662005) q[4];
ry(0.39023614316079214) q[6];
cx q[4],q[6];
ry(1.107192561950922) q[4];
ry(-1.4960326747598656) q[6];
cx q[4],q[6];
ry(-2.564469413468223) q[4];
ry(-2.4391138242322383) q[7];
cx q[4],q[7];
ry(1.7231829661855214) q[4];
ry(-0.08812815462632263) q[7];
cx q[4],q[7];
ry(-1.4051583776161658) q[5];
ry(-0.21886665181135642) q[6];
cx q[5],q[6];
ry(-1.6881928615502888) q[5];
ry(1.2928363287771383) q[6];
cx q[5],q[6];
ry(2.926061634832598) q[5];
ry(2.767639083017943) q[7];
cx q[5],q[7];
ry(-1.9514853121650084) q[5];
ry(1.3985427431915625) q[7];
cx q[5],q[7];
ry(2.9201100358688032) q[6];
ry(2.876272348908007) q[7];
cx q[6],q[7];
ry(1.7453075390643007) q[6];
ry(1.4078838850500883) q[7];
cx q[6],q[7];
ry(2.983415563966904) q[0];
ry(2.283940023329605) q[1];
ry(-3.1054942802852863) q[2];
ry(-1.621740673694643) q[3];
ry(-1.9080481770964965) q[4];
ry(-2.5044329155634375) q[5];
ry(2.395283887962925) q[6];
ry(-2.5997310179038147) q[7];