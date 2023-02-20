OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-0.44399812366575464) q[0];
rz(0.1293242883401895) q[0];
ry(-2.1291119672778756) q[1];
rz(-0.21317942246591579) q[1];
ry(0.571009529711742) q[2];
rz(-0.8507717005860602) q[2];
ry(0.36096413552999884) q[3];
rz(1.477508425823622) q[3];
ry(-3.136841234742416) q[4];
rz(0.409868264677324) q[4];
ry(-0.5842703567184966) q[5];
rz(0.4007186382141734) q[5];
ry(-3.124536749839442) q[6];
rz(1.7897119500515932) q[6];
ry(-2.276762587585406) q[7];
rz(1.3378894428648995) q[7];
ry(1.0773901082868766) q[8];
rz(1.059602784502756) q[8];
ry(-3.1395883800683055) q[9];
rz(0.6398018079613683) q[9];
ry(2.611705534420151) q[10];
rz(0.7092616201502233) q[10];
ry(1.50359681865047) q[11];
rz(1.0980979653621814) q[11];
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
ry(-0.7249329802839782) q[0];
rz(0.44827311337178966) q[0];
ry(-1.5038410898805168) q[1];
rz(0.647225228810787) q[1];
ry(-0.23566873875207153) q[2];
rz(0.5099655424247912) q[2];
ry(-0.9830273409069532) q[3];
rz(-2.761075039420915) q[3];
ry(-3.1382175530259655) q[4];
rz(1.829584103181535) q[4];
ry(1.217047122710098) q[5];
rz(1.2625541675145024) q[5];
ry(1.5842029006696183) q[6];
rz(2.6185819175490037) q[6];
ry(0.021789913173124356) q[7];
rz(-2.234582984894143) q[7];
ry(1.4667685244145132) q[8];
rz(3.0871974945856575) q[8];
ry(0.6564088764067355) q[9];
rz(-1.2128993764805136) q[9];
ry(1.540154704233626) q[10];
rz(2.7794110383335577) q[10];
ry(-2.309369612866605) q[11];
rz(1.3596621881856574) q[11];
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
ry(-2.2755060700420926) q[0];
rz(2.987670667482552) q[0];
ry(2.0841031844381157) q[1];
rz(2.527194505096737) q[1];
ry(0.22907134620060002) q[2];
rz(-2.01143833342649) q[2];
ry(-0.9333903887521854) q[3];
rz(-2.8838513104922034) q[3];
ry(0.7725753659259267) q[4];
rz(-0.8509765288129687) q[4];
ry(0.17771472220183643) q[5];
rz(-2.2386672749640373) q[5];
ry(3.1380282144174414) q[6];
rz(-2.078441997519816) q[6];
ry(-0.016741796170629538) q[7];
rz(0.17508833300986737) q[7];
ry(0.007473122910278462) q[8];
rz(1.8901759412713872) q[8];
ry(3.1388077075026097) q[9];
rz(-0.7225900639780579) q[9];
ry(-3.0210273535893717) q[10];
rz(2.4237062023823635) q[10];
ry(-1.109455762998409) q[11];
rz(1.3035146359933076) q[11];
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
ry(-1.2410297193462776) q[0];
rz(-0.10975414452940213) q[0];
ry(1.6895854307701847) q[1];
rz(-2.4409835064299794) q[1];
ry(-1.1808806440934232) q[2];
rz(-1.4479834354695669) q[2];
ry(2.6787532364927515) q[3];
rz(-3.013921840002567) q[3];
ry(-3.1371223201151692) q[4];
rz(-0.339445459147762) q[4];
ry(-2.6435845497890478) q[5];
rz(-0.6415332567677633) q[5];
ry(-0.5138731228951708) q[6];
rz(-1.2178692322836238) q[6];
ry(1.6819376849456125) q[7];
rz(-1.4721757208231985) q[7];
ry(-2.2517424393744587) q[8];
rz(-0.8091050179390695) q[8];
ry(-1.5229931003042783) q[9];
rz(-3.1056939871312563) q[9];
ry(0.6191774115518597) q[10];
rz(-2.663547380091606) q[10];
ry(-0.31188580185351666) q[11];
rz(-0.8754058398955458) q[11];
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
ry(-2.13329222455312) q[0];
rz(0.04528334076823448) q[0];
ry(1.3581809050849127) q[1];
rz(1.7663223598187476) q[1];
ry(0.5253554488825571) q[2];
rz(-2.580479596864773) q[2];
ry(1.4651927049089177) q[3];
rz(-0.803001322218461) q[3];
ry(0.25201671799077907) q[4];
rz(-2.234754865118642) q[4];
ry(2.732408926715673) q[5];
rz(0.4037276291251146) q[5];
ry(3.1223856145752196) q[6];
rz(1.6426876796498049) q[6];
ry(-3.139714572411739) q[7];
rz(-1.9903121277570164) q[7];
ry(0.0163437474578485) q[8];
rz(0.4312849150926575) q[8];
ry(3.1374975164737013) q[9];
rz(-2.277213701219557) q[9];
ry(-0.8896076562379861) q[10];
rz(-2.789810501570933) q[10];
ry(2.5635103027481607) q[11];
rz(-0.7558229676726587) q[11];
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
ry(2.3609577523792193) q[0];
rz(-2.392013482572396) q[0];
ry(-0.6094256165202502) q[1];
rz(1.7257349838230456) q[1];
ry(0.9579183340838446) q[2];
rz(0.8504420201725741) q[2];
ry(-2.331494618225058) q[3];
rz(-1.7808410429074888) q[3];
ry(-3.132526007174806) q[4];
rz(-1.5099092032998989) q[4];
ry(0.10262663001427057) q[5];
rz(0.4399636230378743) q[5];
ry(-3.091807390548979) q[6];
rz(2.834176651885903) q[6];
ry(2.113362642744528) q[7];
rz(-1.5743474090283787) q[7];
ry(-1.0086399190677815) q[8];
rz(0.8651192278200329) q[8];
ry(-1.3111816955513051) q[9];
rz(2.5797596029331262) q[9];
ry(2.4269998550225447) q[10];
rz(-2.504501028545671) q[10];
ry(-2.539766685748993) q[11];
rz(-0.7375722869885779) q[11];
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
ry(1.3355642734881539) q[0];
rz(1.0086816772115244) q[0];
ry(-1.4264121956223443) q[1];
rz(-1.6769985851770994) q[1];
ry(0.7304566110655726) q[2];
rz(0.5303796182942291) q[2];
ry(-1.5539007058105412) q[3];
rz(0.33702808736827383) q[3];
ry(-1.6818347470394182) q[4];
rz(0.6527517330007065) q[4];
ry(0.7682763505991623) q[5];
rz(2.5225158288738108) q[5];
ry(-0.8410119354346053) q[6];
rz(-0.09441745758138073) q[6];
ry(3.137770856468153) q[7];
rz(0.5712349399970389) q[7];
ry(-3.1258773390575194) q[8];
rz(1.280855762795954) q[8];
ry(-0.24133700097182054) q[9];
rz(-0.914821197577746) q[9];
ry(1.9736356585854322) q[10];
rz(2.910054998271395) q[10];
ry(1.5702765928815028) q[11];
rz(-2.7209939773492104) q[11];
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
ry(-2.9856032533506927) q[0];
rz(2.0023380983720593) q[0];
ry(-1.6790051881994508) q[1];
rz(1.115106599149187) q[1];
ry(1.4615141676027528) q[2];
rz(2.790325927891087) q[2];
ry(1.7064041741130165) q[3];
rz(2.4035209593048226) q[3];
ry(-3.139612819213871) q[4];
rz(1.8390240914365257) q[4];
ry(2.684342730489002) q[5];
rz(1.7405588817996147) q[5];
ry(-3.06649563214274) q[6];
rz(-0.7549216537215078) q[6];
ry(-1.6901721472637279) q[7];
rz(1.8468769637196771) q[7];
ry(-1.3635311895597493) q[8];
rz(1.2095734141961971) q[8];
ry(-1.3904805969608647) q[9];
rz(-2.3852707636176285) q[9];
ry(1.217986297478123) q[10];
rz(1.2361558639869539) q[10];
ry(-1.2150464477199971) q[11];
rz(2.4344064839874724) q[11];
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
ry(1.7975823757973688) q[0];
rz(-2.8940440535883845) q[0];
ry(1.6345940192196364) q[1];
rz(1.8871927319764743) q[1];
ry(-2.023078948095206) q[2];
rz(-2.693200792032621) q[2];
ry(0.8304746872203204) q[3];
rz(2.564275444134517) q[3];
ry(0.3328116993171788) q[4];
rz(-1.183435370542048) q[4];
ry(-3.1357388865460294) q[5];
rz(-0.887271926577035) q[5];
ry(-3.12459476015033) q[6];
rz(-3.020576362181519) q[6];
ry(-0.004554375704984806) q[7];
rz(-0.5048774374573389) q[7];
ry(-3.1214251575260112) q[8];
rz(-2.4369338632531434) q[8];
ry(-1.5493168745294605) q[9];
rz(2.0365570419915304) q[9];
ry(-1.472323193797159) q[10];
rz(-1.1659073879076125) q[10];
ry(-0.0937763256035753) q[11];
rz(2.1415295555657927) q[11];
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
ry(0.4083986149250409) q[0];
rz(0.8427275592493402) q[0];
ry(-1.843434953478896) q[1];
rz(3.074888318978088) q[1];
ry(-0.43654814210720894) q[2];
rz(-1.884055325931657) q[2];
ry(1.1954425276524594) q[3];
rz(-0.08826301954171926) q[3];
ry(-3.1390983341888505) q[4];
rz(-1.9967394390914701) q[4];
ry(2.3466931452431408) q[5];
rz(0.6785006977796226) q[5];
ry(-3.130310159426004) q[6];
rz(1.170454464156457) q[6];
ry(3.137869783667012) q[7];
rz(1.1635914217581194) q[7];
ry(-3.1170367451970615) q[8];
rz(-2.0360229168072825) q[8];
ry(-2.3330404816808796) q[9];
rz(0.5540047307838296) q[9];
ry(-2.360556852144911) q[10];
rz(-2.323295829359393) q[10];
ry(-1.4969373481019543) q[11];
rz(-1.2523658511942095) q[11];
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
ry(-2.966488081486887) q[0];
rz(-2.7423065572915206) q[0];
ry(-2.061917028540416) q[1];
rz(-0.8236463357420848) q[1];
ry(-1.6703960744741753) q[2];
rz(0.6243272825774658) q[2];
ry(2.1276052545160855) q[3];
rz(1.1138972734075923) q[3];
ry(1.1592970271310017) q[4];
rz(0.7694465452764039) q[4];
ry(0.007627675368289566) q[5];
rz(-1.1498152797923837) q[5];
ry(0.014868139182664063) q[6];
rz(2.649474299004979) q[6];
ry(-3.1395154457218464) q[7];
rz(-0.13949437957914196) q[7];
ry(3.1042671973876317) q[8];
rz(-1.9510657461695018) q[8];
ry(-1.7330451621301266) q[9];
rz(-1.9596900630907665) q[9];
ry(1.668859797954717) q[10];
rz(3.129143361254593) q[10];
ry(-0.01571516112589153) q[11];
rz(-2.7776758773956516) q[11];
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
ry(0.4505802036783903) q[0];
rz(0.3010031528799092) q[0];
ry(-2.1340158398007554) q[1];
rz(2.935688577485566) q[1];
ry(-0.12776065337303066) q[2];
rz(2.23477438176006) q[2];
ry(2.9405469878421933) q[3];
rz(1.4704813515615491) q[3];
ry(0.0043621123660457775) q[4];
rz(-1.5441025909364248) q[4];
ry(1.0036706571146445) q[5];
rz(-3.0145933343149545) q[5];
ry(0.019985132182600118) q[6];
rz(-1.887576263890056) q[6];
ry(-1.563583304421016) q[7];
rz(2.4033484334476896) q[7];
ry(-3.028921844575478) q[8];
rz(-2.011621813480609) q[8];
ry(-2.9543968895361243) q[9];
rz(1.3757756596795305) q[9];
ry(0.7958874866389616) q[10];
rz(2.6404837032277335) q[10];
ry(0.4577292247480061) q[11];
rz(-2.3056267671290023) q[11];
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
ry(-2.6682254467734894) q[0];
rz(-2.1956141632522383) q[0];
ry(-1.8028373335976193) q[1];
rz(2.776077692183253) q[1];
ry(-0.1230934310865295) q[2];
rz(2.075297591414934) q[2];
ry(-1.5141722492274792) q[3];
rz(1.390346772538328) q[3];
ry(2.1185575838238195) q[4];
rz(1.6680477372351405) q[4];
ry(3.1415351668126394) q[5];
rz(1.0381323861546212) q[5];
ry(1.571079698906039) q[6];
rz(-2.5717141510431465) q[6];
ry(-0.0015340929266183887) q[7];
rz(-1.244870821268348) q[7];
ry(-0.047450574619033674) q[8];
rz(0.8087443371718568) q[8];
ry(1.3991090051356716) q[9];
rz(3.010981534134606) q[9];
ry(-2.621780651484939) q[10];
rz(-1.928312865772284) q[10];
ry(1.6457099296247122) q[11];
rz(3.1371163715738346) q[11];
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
ry(-1.9565900048939393) q[0];
rz(-1.7464527870617292) q[0];
ry(0.49944026532677155) q[1];
rz(0.31575258857021715) q[1];
ry(0.2941016281838582) q[2];
rz(-1.4717644976522646) q[2];
ry(1.4398506338556019) q[3];
rz(-0.5888428766412677) q[3];
ry(3.1411235433369713) q[4];
rz(-0.27617724955733275) q[4];
ry(-1.2261301667136386) q[5];
rz(1.2172817300439644) q[5];
ry(3.1414925434727046) q[6];
rz(-1.0777062929951935) q[6];
ry(-3.1082835075896043) q[7];
rz(1.1663054368186971) q[7];
ry(0.033855005651454384) q[8];
rz(-0.14540079158216385) q[8];
ry(1.5706752448538117) q[9];
rz(1.0828498651053597) q[9];
ry(-2.11152048263489) q[10];
rz(-2.6179889776634697) q[10];
ry(1.901522358719049) q[11];
rz(0.024276623926056114) q[11];
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
ry(-1.090768519372701) q[0];
rz(1.6087844025116567) q[0];
ry(3.07014815309542) q[1];
rz(-1.7998977804033254) q[1];
ry(0.9135748498026787) q[2];
rz(1.83827643963372) q[2];
ry(-3.009088845286309) q[3];
rz(-0.3073065584843184) q[3];
ry(-1.909499395876928) q[4];
rz(-3.091464744956621) q[4];
ry(3.140096882624182) q[5];
rz(0.3535600929475491) q[5];
ry(1.17511075226257) q[6];
rz(1.547160026062241) q[6];
ry(0.27820393410265787) q[7];
rz(2.9945484196243286) q[7];
ry(-3.1273181241403187) q[8];
rz(-2.679392440837062) q[8];
ry(-3.0478743953558873) q[9];
rz(-0.2857734962527663) q[9];
ry(-0.8160158908032358) q[10];
rz(0.48320600599505314) q[10];
ry(1.5866250486020015) q[11];
rz(-1.5554380047214138) q[11];
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
ry(0.6309988284353816) q[0];
rz(-1.3047461194422325) q[0];
ry(1.970325567828912) q[1];
rz(2.1177813504011214) q[1];
ry(-1.0776990132410358) q[2];
rz(0.0218006749259807) q[2];
ry(-2.4541610961685154) q[3];
rz(1.786950867753911) q[3];
ry(-0.0014815618375179085) q[4];
rz(-0.40348680394040315) q[4];
ry(-0.06730736951129587) q[5];
rz(2.3194887858427555) q[5];
ry(3.121558510165877) q[6];
rz(-1.175667152208268) q[6];
ry(0.06981341644340579) q[7];
rz(0.06501028472550668) q[7];
ry(-1.5965177080008595) q[8];
rz(-1.466316205382741) q[8];
ry(1.6920639234696107) q[9];
rz(1.623151493742764) q[9];
ry(1.4340989151642578) q[10];
rz(0.2906714067703504) q[10];
ry(1.5805439577432976) q[11];
rz(-1.2879761751692804) q[11];
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
ry(0.656590210669798) q[0];
rz(-0.5741786202597857) q[0];
ry(-0.45517475470987867) q[1];
rz(0.35665286920641653) q[1];
ry(-0.6474226178122231) q[2];
rz(-2.2334524599079244) q[2];
ry(0.6620539136249576) q[3];
rz(-2.4119983295486023) q[3];
ry(1.934089232289618) q[4];
rz(-0.7419950732849588) q[4];
ry(-0.0010534057442117052) q[5];
rz(-1.289205868613963) q[5];
ry(-2.4981893359564653) q[6];
rz(1.3906778948313425) q[6];
ry(-3.100050592910281) q[7];
rz(1.5264920233707917) q[7];
ry(0.010829480298849859) q[8];
rz(2.569432349538742) q[8];
ry(-1.5313280918403296) q[9];
rz(-2.4004478142225896) q[9];
ry(-2.993306618035939) q[10];
rz(-0.04584733553992403) q[10];
ry(1.146867725594229) q[11];
rz(-1.0788980353307887) q[11];
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
ry(2.0843895278063274) q[0];
rz(1.037417793916821) q[0];
ry(-2.6752480097825746) q[1];
rz(1.4746650712954077) q[1];
ry(-2.90270322292493) q[2];
rz(-0.27049055539401046) q[2];
ry(2.2701364543684126) q[3];
rz(1.931019527774338) q[3];
ry(3.140449191046562) q[4];
rz(1.3477980852539286) q[4];
ry(-2.406715410029058) q[5];
rz(-1.9771841015355252) q[5];
ry(3.113062920352945) q[6];
rz(-1.9843910203066217) q[6];
ry(-2.653764354471899) q[7];
rz(-2.700652535053004) q[7];
ry(3.1254519663365166) q[8];
rz(-1.1235155103227212) q[8];
ry(-2.8765429601685075) q[9];
rz(-1.688778374804845) q[9];
ry(1.4779750816895145) q[10];
rz(2.6850544674412844) q[10];
ry(-0.5731349273180628) q[11];
rz(-3.1393133089527328) q[11];
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
ry(-2.554861083619641) q[0];
rz(-2.9258986311514152) q[0];
ry(-1.1251716481124934) q[1];
rz(-2.671056284771111) q[1];
ry(-2.8975751528211426) q[2];
rz(-2.8160362568854844) q[2];
ry(2.547602287525949) q[3];
rz(1.8962242674756038) q[3];
ry(-3.1374087081712743) q[4];
rz(0.8396380438972955) q[4];
ry(-0.0043497243344789585) q[5];
rz(1.5455227893106025) q[5];
ry(2.312735205513602) q[6];
rz(-1.3082265034966583) q[6];
ry(3.1392890041496653) q[7];
rz(2.2869283123262996) q[7];
ry(-3.1248082476286103) q[8];
rz(-3.102911755080406) q[8];
ry(-1.4975493050716442) q[9];
rz(3.0961152103819773) q[9];
ry(1.4507212572565609) q[10];
rz(-0.5530954460501967) q[10];
ry(-0.843817252189785) q[11];
rz(-2.754349248894596) q[11];
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
ry(2.9949788017062966) q[0];
rz(0.6068527344820591) q[0];
ry(1.2460519349686972) q[1];
rz(-3.046718848544693) q[1];
ry(1.1444290650426199) q[2];
rz(1.8996948992665503) q[2];
ry(-1.4116114433372804) q[3];
rz(2.7244016443939425) q[3];
ry(-0.0015156399160715685) q[4];
rz(-2.339728661883499) q[4];
ry(0.6911000958431117) q[5];
rz(0.5449819453056169) q[5];
ry(0.11547453013777442) q[6];
rz(-1.241907837595392) q[6];
ry(-3.1381433504520952) q[7];
rz(-0.4315645571192839) q[7];
ry(-3.1410655720298966) q[8];
rz(2.2654706726344167) q[8];
ry(-2.6552184279105684) q[9];
rz(-0.027298148793394535) q[9];
ry(0.831868164997765) q[10];
rz(1.8028252079235543) q[10];
ry(3.130669374441767) q[11];
rz(1.4781099841715004) q[11];
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
ry(0.5668649192126868) q[0];
rz(1.4091566039739085) q[0];
ry(0.06562294342242632) q[1];
rz(0.8101178380086362) q[1];
ry(1.3064095028174627) q[2];
rz(-1.6454236142578011) q[2];
ry(1.5676918258420312) q[3];
rz(2.5344195323357894) q[3];
ry(-3.031763868556888) q[4];
rz(-0.9496893349118581) q[4];
ry(3.140019896672233) q[5];
rz(-0.08397222678033872) q[5];
ry(-0.16450209132809948) q[6];
rz(-2.4119027325079543) q[6];
ry(-0.0022896754502441477) q[7];
rz(2.3204866480982207) q[7];
ry(-3.12888216925745) q[8];
rz(-0.6340533377446965) q[8];
ry(1.4674927257612636) q[9];
rz(-3.0357551069993884) q[9];
ry(0.09610217478134754) q[10];
rz(-0.23058496472964152) q[10];
ry(-1.005528214373145) q[11];
rz(0.9878506030343861) q[11];
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
ry(-1.0362205641021935) q[0];
rz(-1.3424822548479387) q[0];
ry(-3.112697157136528) q[1];
rz(1.0257889161978633) q[1];
ry(-1.0786077140879562) q[2];
rz(0.36371529003640113) q[2];
ry(1.5466124455932375) q[3];
rz(1.5291322483959073) q[3];
ry(3.1381000704950437) q[4];
rz(0.46264934267626495) q[4];
ry(1.6747675467887324) q[5];
rz(0.18155397827444544) q[5];
ry(1.7412818760229856) q[6];
rz(-1.3568545254064666) q[6];
ry(1.9677320493723016) q[7];
rz(-0.9954385450417172) q[7];
ry(1.4783246007288378) q[8];
rz(-0.9911576901895635) q[8];
ry(0.004281656073978505) q[9];
rz(-2.6696930122086933) q[9];
ry(1.7648380345345365) q[10];
rz(0.25420906183007674) q[10];
ry(0.23846670379184287) q[11];
rz(2.624084343744155) q[11];
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
ry(-2.535697307898068) q[0];
rz(2.440768556324739) q[0];
ry(-1.242360141181913) q[1];
rz(-0.5052703499338983) q[1];
ry(-1.959511069081544) q[2];
rz(1.4391718904564055) q[2];
ry(2.761419732750986) q[3];
rz(3.040202569926162) q[3];
ry(3.1379223520160235) q[4];
rz(0.34366309967602815) q[4];
ry(3.122629889354699) q[5];
rz(1.9926355982464488) q[5];
ry(-2.826280957998501) q[6];
rz(1.9950194354168964) q[6];
ry(0.006701005492525229) q[7];
rz(-2.1744281769112304) q[7];
ry(0.00018425501260477262) q[8];
rz(2.792358503107826) q[8];
ry(0.023698869539555804) q[9];
rz(2.028069996133668) q[9];
ry(1.5510227151492688) q[10];
rz(-1.6053756157407841) q[10];
ry(1.2204468571726539) q[11];
rz(-0.5210763502000002) q[11];
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
ry(3.023583535123823) q[0];
rz(-1.191059753618539) q[0];
ry(-0.9487343674651999) q[1];
rz(1.013961256950715) q[1];
ry(0.28265721887985773) q[2];
rz(-0.4468006156127587) q[2];
ry(1.8521421239428495) q[3];
rz(0.10238868719854731) q[3];
ry(-0.0030374219913956112) q[4];
rz(1.5780612775718075) q[4];
ry(1.1980562218439283) q[5];
rz(2.9540542653839723) q[5];
ry(1.7695002784073368) q[6];
rz(1.852741586485525) q[6];
ry(1.582038944689593) q[7];
rz(0.7941496355421985) q[7];
ry(1.5625953431797983) q[8];
rz(-2.2659193343249218) q[8];
ry(-3.125212646534782) q[9];
rz(2.6449652539462885) q[9];
ry(1.6859450858769465) q[10];
rz(1.3101355073997165) q[10];
ry(-3.1010170725209174) q[11];
rz(-1.1537877507534484) q[11];
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
ry(-2.535766489088002) q[0];
rz(-0.6045496572727354) q[0];
ry(-2.3608126829959364) q[1];
rz(-0.8795878194635119) q[1];
ry(1.4605958877936978) q[2];
rz(-2.1521619134911667) q[2];
ry(-1.6889930448587558) q[3];
rz(0.01174639448426973) q[3];
ry(-0.05264755173126012) q[4];
rz(0.35475021827377345) q[4];
ry(1.5695949951287247) q[5];
rz(1.1978368240468174) q[5];
ry(0.010178442439376028) q[6];
rz(-2.926646977781049) q[6];
ry(-0.002085093020040673) q[7];
rz(1.4348510475660678) q[7];
ry(-3.139015863441191) q[8];
rz(-0.7889513014811591) q[8];
ry(0.018418298359579713) q[9];
rz(1.018042127080479) q[9];
ry(-1.4912670506579966) q[10];
rz(-1.9236026349503057) q[10];
ry(1.6983071260525782) q[11];
rz(-0.5130216328383782) q[11];
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
ry(2.04620060338377) q[0];
rz(0.3742307405641086) q[0];
ry(1.4658219847953917) q[1];
rz(-1.5791497468650746) q[1];
ry(1.903964722600639) q[2];
rz(1.5164125857372124) q[2];
ry(-1.567464703420681) q[3];
rz(2.872045908572682) q[3];
ry(0.26315056205453224) q[4];
rz(2.230051350329562) q[4];
ry(3.140197351054046) q[5];
rz(1.1963623104188699) q[5];
ry(2.953762500436699) q[6];
rz(-1.5202945285767981) q[6];
ry(-0.003507958368126435) q[7];
rz(2.3077212667067504) q[7];
ry(1.6117563189447004) q[8];
rz(0.5838617421914954) q[8];
ry(-1.667491928504913) q[9];
rz(0.8076797707649049) q[9];
ry(-2.522776887855874) q[10];
rz(-0.4180329239222411) q[10];
ry(1.5462917113233088) q[11];
rz(-1.3996854568727384) q[11];
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
ry(-1.8502745839911094) q[0];
rz(2.6592212631546643) q[0];
ry(-0.0008417786239656184) q[1];
rz(-0.09772199606344234) q[1];
ry(1.5708589216997082) q[2];
rz(0.0008034214442057991) q[2];
ry(3.1066877672441766) q[3];
rz(-1.771556118664937) q[3];
ry(0.01738983078538441) q[4];
rz(-1.1765264172242116) q[4];
ry(-1.549840397300521) q[5];
rz(0.8897929079515816) q[5];
ry(0.009903687581640043) q[6];
rz(-2.6401598433637306) q[6];
ry(-3.1237920155769325) q[7];
rz(-0.2722510526996076) q[7];
ry(-0.01629113351599393) q[8];
rz(0.4050485259331431) q[8];
ry(-3.0984445102857676) q[9];
rz(2.161890198919294) q[9];
ry(-2.1638438069070833) q[10];
rz(0.4184718188589818) q[10];
ry(2.7251424868218836) q[11];
rz(2.8727382255757647) q[11];
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
ry(-2.5671418945319875) q[0];
rz(-0.015693658738978968) q[0];
ry(1.4989732730936804) q[1];
rz(-2.2667454432971805) q[1];
ry(1.223973303032477) q[2];
rz(-1.3739552883017323) q[2];
ry(1.5676244102311792) q[3];
rz(-1.5427471099014372) q[3];
ry(-1.6947710775171354) q[4];
rz(2.40079364666708) q[4];
ry(0.0028020980657273084) q[5];
rz(1.681839674436927) q[5];
ry(0.003253661919163763) q[6];
rz(2.6800017425957883) q[6];
ry(-3.0746238614132357) q[7];
rz(-0.27752962142521276) q[7];
ry(-3.1351952235477167) q[8];
rz(2.4769607829389577) q[8];
ry(0.07746155859317985) q[9];
rz(-0.1745144937541323) q[9];
ry(-2.5460770575510843) q[10];
rz(2.2805407697852877) q[10];
ry(-1.3506894830253215) q[11];
rz(0.036726199393130265) q[11];
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
ry(-2.3246963020594813) q[0];
rz(-1.3610145517069174) q[0];
ry(-0.022298047734068314) q[1];
rz(-0.7609075347528477) q[1];
ry(3.136012002229623) q[2];
rz(-2.646186586643792) q[2];
ry(1.5679745307072899) q[3];
rz(1.1698480295203448) q[3];
ry(3.13668935806341) q[4];
rz(-2.409987484404474) q[4];
ry(-3.13655341094886) q[5];
rz(-2.2414157294687795) q[5];
ry(3.138627593729071) q[6];
rz(-0.26412971231323296) q[6];
ry(3.131666728477512) q[7];
rz(-0.0796119411147927) q[7];
ry(-0.010135635079306088) q[8];
rz(-0.515913261477866) q[8];
ry(0.02856144329705934) q[9];
rz(2.9422372101146124) q[9];
ry(1.6241809164162313) q[10];
rz(0.8992657053907606) q[10];
ry(-1.138842084753592) q[11];
rz(2.532518611785771) q[11];
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
ry(-1.3175534447353296) q[0];
rz(1.5813097372800664) q[0];
ry(1.6734853772648715) q[1];
rz(0.9651544752307637) q[1];
ry(0.35813422969566927) q[2];
rz(2.308589266814746) q[2];
ry(-0.08708288167542833) q[3];
rz(2.930022843055447) q[3];
ry(3.00470780196728) q[4];
rz(2.4762501244657034) q[4];
ry(1.559166457621168) q[5];
rz(-0.6073712869706389) q[5];
ry(2.419344187466523) q[6];
rz(-0.9793275137642583) q[6];
ry(-0.16910811546141144) q[7];
rz(-2.274422452569076) q[7];
ry(2.1373837765794153) q[8];
rz(0.3196850645496356) q[8];
ry(-1.1841106808667712) q[9];
rz(1.2203034212394928) q[9];
ry(2.016147996688491) q[10];
rz(2.769212226751012) q[10];
ry(-1.0983637408858886) q[11];
rz(1.2989742922529803) q[11];