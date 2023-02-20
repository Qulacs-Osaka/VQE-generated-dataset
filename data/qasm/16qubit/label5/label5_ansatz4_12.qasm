OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(1.8067806874026937) q[0];
rz(-2.3169226213094114) q[0];
ry(-3.0674215978282526) q[1];
rz(-2.443668367323746) q[1];
ry(-2.718403348123403) q[2];
rz(-2.417297506323539) q[2];
ry(-0.00014784366374165359) q[3];
rz(1.453689871768324) q[3];
ry(-1.5692654703901505) q[4];
rz(-1.574497202722819) q[4];
ry(1.5603790529915784) q[5];
rz(0.45267018476064436) q[5];
ry(-1.5706096883464284) q[6];
rz(1.5726220604269152) q[6];
ry(-1.6525655942495527) q[7];
rz(-1.3315460175836291) q[7];
ry(1.5707225204785242) q[8];
rz(3.1414960489706125) q[8];
ry(-1.570832797288064) q[9];
rz(-2.9203791405183335) q[9];
ry(-3.1415106993045843) q[10];
rz(1.524979710420017) q[10];
ry(-0.0013413978917871106) q[11];
rz(-0.8978669939994756) q[11];
ry(1.5602817132164708) q[12];
rz(-2.6589194979122155) q[12];
ry(0.0013813530413244732) q[13];
rz(2.612358572087386) q[13];
ry(-0.0001137281696426129) q[14];
rz(0.5732801158831534) q[14];
ry(1.5681817056246266) q[15];
rz(-2.338600088872375) q[15];
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
ry(-3.031712356841029) q[0];
rz(0.8243297176092272) q[0];
ry(-1.7344432316706389) q[1];
rz(0.9733805745289842) q[1];
ry(-3.1415440296375166) q[2];
rz(-1.8028148719178159) q[2];
ry(-3.138148789852859) q[3];
rz(2.0342722378754603) q[3];
ry(1.5727183973787895) q[4];
rz(-1.025791535559479) q[4];
ry(-1.8963109303271608e-05) q[5];
rz(-2.0221943529254967) q[5];
ry(-2.645408681584011) q[6];
rz(-0.507986077011663) q[6];
ry(-3.138283010210302) q[7];
rz(-2.9023931380458285) q[7];
ry(-2.4048259291931013) q[8];
rz(-3.134108423235536) q[8];
ry(0.7488389608551405) q[9];
rz(1.406753065032066) q[9];
ry(-1.5714665704881785) q[10];
rz(0.334074879018049) q[10];
ry(-2.069903112686995) q[11];
rz(-0.3277374368835735) q[11];
ry(-0.3488688562933904) q[12];
rz(1.2879110378460508) q[12];
ry(1.5883131190499071) q[13];
rz(0.3683141286455848) q[13];
ry(1.578649127028081) q[14];
rz(1.4569874206636808) q[14];
ry(2.764405381764597) q[15];
rz(2.6180109445913207) q[15];
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
ry(1.57861483154515) q[0];
rz(-2.931837578413973) q[0];
ry(1.074838843364061) q[1];
rz(3.0777445466933897) q[1];
ry(0.0010069117570061792) q[2];
rz(2.963520954517335) q[2];
ry(-3.1304472492683875) q[3];
rz(1.625728137913372) q[3];
ry(-3.1409081434351176) q[4];
rz(-1.4732980524682464) q[4];
ry(-1.614431801900385) q[5];
rz(3.1415523012715365) q[5];
ry(-0.0026467327564647647) q[6];
rz(-2.644545055895426) q[6];
ry(-1.5246294362332078) q[7];
rz(0.0031526586064931222) q[7];
ry(1.0404140545891918) q[8];
rz(-1.5564652767152607) q[8];
ry(-3.13310734554155) q[9];
rz(0.0008480860224144849) q[9];
ry(-6.586192808580009e-05) q[10];
rz(-1.287683218543793) q[10];
ry(3.1415274539279996) q[11];
rz(-2.9769211276982412) q[11];
ry(1.3969227839533627) q[12];
rz(1.328007119906231) q[12];
ry(-1.9657285426875193) q[13];
rz(0.931319074167048) q[13];
ry(0.027817054165534923) q[14];
rz(2.968558412838525) q[14];
ry(2.264888757349523) q[15];
rz(2.1100384183429997) q[15];
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
ry(1.4587495250869535) q[0];
rz(2.9847001871888943) q[0];
ry(-1.697046274359574) q[1];
rz(-1.172237599405749) q[1];
ry(-1.5707599472096767) q[2];
rz(2.4477233905583238) q[2];
ry(1.5708134264464872) q[3];
rz(-0.9422002894762793) q[3];
ry(-0.0021751923108777054) q[4];
rz(0.447192150234588) q[4];
ry(-1.5832249434081975) q[5];
rz(1.569859554117432) q[5];
ry(-0.24964842375805982) q[6];
rz(0.2650233820694133) q[6];
ry(0.6530617338749174) q[7];
rz(-0.6745310268858677) q[7];
ry(1.6363291867477407) q[8];
rz(-7.877642054757836e-05) q[8];
ry(-1.5767305550523458) q[9];
rz(-2.8046858217559136) q[9];
ry(3.1360106325439987) q[10];
rz(-0.8032641503259539) q[10];
ry(-3.1410850257759524) q[11];
rz(-1.3976226466668413) q[11];
ry(-3.138744239909416) q[12];
rz(3.0702738025300857) q[12];
ry(0.005503836597975575) q[13];
rz(2.987104316839919) q[13];
ry(-0.00663089680223499) q[14];
rz(1.1695901771429076) q[14];
ry(-0.0028788605212577423) q[15];
rz(-2.982608726249984) q[15];
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
ry(-1.5750046215712747) q[0];
rz(2.7880341867197527) q[0];
ry(-1.5703062591892196) q[1];
rz(-1.568146021328916) q[1];
ry(3.1413793688663483) q[2];
rz(2.435234957818462) q[2];
ry(-0.0012317424826541186) q[3];
rz(-0.587041829781823) q[3];
ry(-1.5709315924790204) q[4];
rz(2.33175718129647) q[4];
ry(1.5713483228753429) q[5];
rz(-2.8183471717609514) q[5];
ry(-3.1362574344354255) q[6];
rz(-1.3186815976484865) q[6];
ry(-3.1398846426308755) q[7];
rz(0.900321663734036) q[7];
ry(-0.6423454077971691) q[8];
rz(-1.5695093517857963) q[8];
ry(-0.00034519124577777835) q[9];
rz(2.9645329520649506) q[9];
ry(0.008726327288791786) q[10];
rz(1.4201466988246347) q[10];
ry(-3.120755424329688) q[11];
rz(-0.9685720303452321) q[11];
ry(-1.1479195729734988) q[12];
rz(-1.7587451000951717) q[12];
ry(-2.61307521494887) q[13];
rz(-2.3991245546633277) q[13];
ry(-0.03996244543886628) q[14];
rz(-2.460798165045172) q[14];
ry(-1.5090949393145667) q[15];
rz(0.24226134035677183) q[15];
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
ry(-7.402111534302101e-05) q[0];
rz(-1.0966000933641558) q[0];
ry(2.0096696997405505) q[1];
rz(-0.6479973338889131) q[1];
ry(-0.8416856736851951) q[2];
rz(1.585314637231452) q[2];
ry(0.5615094825289999) q[3];
rz(3.1065001698689705) q[3];
ry(0.01941419392123992) q[4];
rz(2.246181534561934) q[4];
ry(3.1281271053693804) q[5];
rz(-2.8184504789929146) q[5];
ry(-1.570805136848545) q[6];
rz(-3.0276903316631185) q[6];
ry(1.5710217222296612) q[7];
rz(3.127369128066276) q[7];
ry(1.5542548081815075) q[8];
rz(3.0769714503040704) q[8];
ry(-0.012725988101169428) q[9];
rz(2.9816042887575103) q[9];
ry(1.5693806878876957) q[10];
rz(0.5846514204973736) q[10];
ry(-0.00015298305868149953) q[11];
rz(1.6069902493336201) q[11];
ry(0.40528640272184685) q[12];
rz(1.5542725818427465) q[12];
ry(1.5694640975760397) q[13];
rz(-1.5547288872666964) q[13];
ry(-1.5716245165451728) q[14];
rz(2.3316767927477056) q[14];
ry(-3.137641020215146) q[15];
rz(-1.4644874199591675) q[15];
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
ry(-0.004640389046915886) q[0];
rz(-3.0031872445170977) q[0];
ry(-1.5715678690988912) q[1];
rz(-1.5517869919487783) q[1];
ry(-1.5707601343057878) q[2];
rz(0.9259374503649429) q[2];
ry(1.566253320408951) q[3];
rz(-1.5718692799395617) q[3];
ry(3.140997616906991) q[4];
rz(-0.1343252295278155) q[4];
ry(-1.5725941968392736) q[5];
rz(1.570090887817913) q[5];
ry(1.527408928867124) q[6];
rz(-2.7517488844102136) q[6];
ry(1.5739091923464767) q[7];
rz(-0.5733203356219665) q[7];
ry(1.5708174255622713) q[8];
rz(-2.961752795236176) q[8];
ry(1.5707864436400247) q[9];
rz(-2.182548950256259) q[9];
ry(0.04282886291747614) q[10];
rz(-2.4048296560449014) q[10];
ry(-3.125614789750313) q[11];
rz(-0.6238957353128817) q[11];
ry(1.5701336902071672) q[12];
rz(2.414494601606249) q[12];
ry(1.5656306289087265) q[13];
rz(-0.11146346783443362) q[13];
ry(-3.0578468194028354) q[14];
rz(0.10857369843694657) q[14];
ry(-1.5008273062509985) q[15];
rz(-1.4985989039482757) q[15];
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
ry(-3.141064023129903) q[0];
rz(-3.134688240822848) q[0];
ry(-3.083803963813993) q[1];
rz(-3.1241389457334283) q[1];
ry(0.003607958895496921) q[2];
rz(0.6446780721884642) q[2];
ry(3.132716773380415) q[3];
rz(0.01566229855797019) q[3];
ry(-1.5709992589433224) q[4];
rz(-1.570368496571296) q[4];
ry(1.5708825938480375) q[5];
rz(1.5717861194525362) q[5];
ry(0.06243398159450031) q[6];
rz(0.3759036345383626) q[6];
ry(1.714285575915297) q[7];
rz(0.0008386412718026648) q[7];
ry(3.141367815482279) q[8];
rz(-1.3513953145882567) q[8];
ry(3.141553521285232) q[9];
rz(2.5451763556060354) q[9];
ry(7.64876594758176e-05) q[10];
rz(-1.3207634022093426) q[10];
ry(-3.141431583065104) q[11];
rz(-1.5100623475357802) q[11];
ry(-0.00028609966431341683) q[12];
rz(2.146753256511679) q[12];
ry(-3.140641079400653) q[13];
rz(-0.9920655320202535) q[13];
ry(1.5640561090916583) q[14];
rz(-2.929390576822555) q[14];
ry(3.042591556272062) q[15];
rz(-0.7733553183866146) q[15];
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
ry(0.005397429470121118) q[0];
rz(-2.513722787935008) q[0];
ry(-0.6497514758946843) q[1];
rz(-3.1398342658494256) q[1];
ry(-1.5946553565540302) q[2];
rz(2.444140817033058) q[2];
ry(1.5722757186097844) q[3];
rz(0.27401329188340734) q[3];
ry(1.5707276386116886) q[4];
rz(-2.0270137747557273) q[4];
ry(-1.5713270767999825) q[5];
rz(-2.658551143263777) q[5];
ry(-1.5925381818616144) q[6];
rz(-1.9544626129856189) q[6];
ry(2.1076053022173697) q[7];
rz(3.0824187445386113) q[7];
ry(-9.428734717253917e-05) q[8];
rz(-1.610978929771195) q[8];
ry(-1.571275845494536) q[9];
rz(-1.5726393140894241) q[9];
ry(-1.5346982732784906) q[10];
rz(1.7046905265012917) q[10];
ry(1.5811399232510643) q[11];
rz(-1.2297965509741304) q[11];
ry(-2.171267157161014) q[12];
rz(1.6137405820455788) q[12];
ry(3.139501188763021) q[13];
rz(1.3774600888316666) q[13];
ry(0.08520925985119024) q[14];
rz(2.752395270882222) q[14];
ry(-1.2798173047592531) q[15];
rz(0.09021131931426996) q[15];
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
ry(0.0002978497672287972) q[0];
rz(-1.582631166576573) q[0];
ry(-1.571101420282632) q[1];
rz(0.46846677223267896) q[1];
ry(3.1362207828912974) q[2];
rz(0.8721053290963799) q[2];
ry(0.05726863336780414) q[3];
rz(-1.7385376302897682) q[3];
ry(-3.1415739808577814) q[4];
rz(1.1748206670869292) q[4];
ry(3.1410456356902894) q[5];
rz(-1.0873738622204745) q[5];
ry(0.00021962462902518033) q[6];
rz(1.9417771301315752) q[6];
ry(1.5729474523014504) q[7];
rz(-0.5167421275661288) q[7];
ry(1.6753851161128432) q[8];
rz(2.7498575672456727) q[8];
ry(-1.726188164931087) q[9];
rz(-1.5903418891397303) q[9];
ry(-3.141496127275508) q[10];
rz(-1.9286315181691727) q[10];
ry(3.1415813209259857) q[11];
rz(-1.225624452472866) q[11];
ry(-3.105279426251432) q[12];
rz(1.8740243151347933) q[12];
ry(-3.136086877207429) q[13];
rz(0.620189915273338) q[13];
ry(-3.1389189176907326) q[14];
rz(-2.8523994383206754) q[14];
ry(-3.11164451075288) q[15];
rz(2.2644816300576815) q[15];
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
ry(-3.1376081209823683) q[0];
rz(-2.850736479653441) q[0];
ry(0.0010434541023037656) q[1];
rz(-1.8797623373402699) q[1];
ry(1.5155532985703672) q[2];
rz(-0.6464275008446699) q[2];
ry(-1.550333259548931) q[3];
rz(1.558513953796865) q[3];
ry(6.935413948205849e-05) q[4];
rz(1.9046396826592265) q[4];
ry(-1.570711020977586) q[5];
rz(2.72103360998179) q[5];
ry(-3.141580792597487) q[6];
rz(1.5900471193562797) q[6];
ry(3.1389727891875636) q[7];
rz(2.348807900702932) q[7];
ry(3.141581690193836) q[8];
rz(2.7500715900200285) q[8];
ry(-0.0037144126930686023) q[9];
rz(-1.5519747710838394) q[9];
ry(-7.162423287388431e-05) q[10];
rz(-2.1528702870258574) q[10];
ry(0.027243995748993832) q[11];
rz(0.11423723892284698) q[11];
ry(2.5264738635030306) q[12];
rz(-2.214112387890858) q[12];
ry(0.006803859055163564) q[13];
rz(0.06681234502221366) q[13];
ry(0.0014986316938392366) q[14];
rz(0.7391573617775579) q[14];
ry(-1.0981479915376289) q[15];
rz(-2.140147089431212) q[15];
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
ry(0.0061115930633978315) q[0];
rz(0.07451116392790953) q[0];
ry(-0.0002711457378600869) q[1];
rz(-1.688286851998198) q[1];
ry(-3.1343112095642978) q[2];
rz(-0.3012004403231119) q[2];
ry(1.5712004517562113) q[3];
rz(3.0270309663919215) q[3];
ry(0.0006555637510201606) q[4];
rz(0.7222555584551469) q[4];
ry(0.00230868087951011) q[5];
rz(-1.4680503027014637) q[5];
ry(1.5706544700490397) q[6];
rz(1.563884087571565) q[6];
ry(0.0043778244628560605) q[7];
rz(0.2762606059634545) q[7];
ry(-1.6755390547892786) q[8];
rz(-1.5664957957766035) q[8];
ry(1.5967237541381962) q[9];
rz(-1.5684332475078353) q[9];
ry(0.0001710614381362774) q[10];
rz(-0.5256172323637078) q[10];
ry(-3.141564671469709) q[11];
rz(2.7237233352896912) q[11];
ry(-3.1331210800488942) q[12];
rz(0.7599041976935769) q[12];
ry(1.5737478343855364) q[13];
rz(-2.38626897460765) q[13];
ry(-1.3500579313224526) q[14];
rz(-0.041769033752254714) q[14];
ry(1.6007091298898044) q[15];
rz(-2.0819277496959874) q[15];
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
ry(-1.570573335127266) q[0];
rz(1.5737030456490144) q[0];
ry(-0.09630346686688096) q[1];
rz(-1.6136144743674126) q[1];
ry(-8.644249477551824e-05) q[2];
rz(-0.3604803833565536) q[2];
ry(0.039193703005782426) q[3];
rz(-3.0518772669708203) q[3];
ry(0.0007648950877232608) q[4];
rz(-1.629917229439161) q[4];
ry(0.00020251136863258523) q[5];
rz(0.3860256406387555) q[5];
ry(-2.001727586941861) q[6];
rz(-1.4083680792485067) q[6];
ry(1.5729308713416639) q[7];
rz(-2.955529015247371) q[7];
ry(-1.5710865720684086) q[8];
rz(-1.9833166540324116) q[8];
ry(-1.5711988166213176) q[9];
rz(-0.00010628957505964819) q[9];
ry(-1.5701677420612365) q[10];
rz(-2.991286203252493) q[10];
ry(3.1414846712687514) q[11];
rz(1.0430294586901532) q[11];
ry(-3.1379064469122318) q[12];
rz(-1.525872654662417) q[12];
ry(-0.0007793983595217764) q[13];
rz(-0.7426182556319633) q[13];
ry(-1.570501731767469) q[14];
rz(1.6309795018714492) q[14];
ry(0.0001653527412196869) q[15];
rz(-1.0710075122655738) q[15];
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
ry(-1.5710048379558235) q[0];
rz(1.5764090103662882) q[0];
ry(-1.5707656006657102) q[1];
rz(1.4687364031847978) q[1];
ry(-1.570785953065237) q[2];
rz(-1.559735855367136) q[2];
ry(-3.1383567426261236) q[3];
rz(3.112987324669538) q[3];
ry(-3.1404281046838878) q[4];
rz(1.9965631239434805) q[4];
ry(-2.389563681148843e-05) q[5];
rz(0.8551836862553227) q[5];
ry(0.0010775965910481133) q[6];
rz(1.4089085014332898) q[6];
ry(-3.14002999148969) q[7];
rz(0.186716710253509) q[7];
ry(0.00017577152736336643) q[8];
rz(1.98332594929008) q[8];
ry(-2.9607202607135683) q[9];
rz(-2.440447016649472e-05) q[9];
ry(3.141534539842738) q[10];
rz(1.762831494863983) q[10];
ry(0.20681325537912887) q[11];
rz(-2.711203982295971) q[11];
ry(1.570353329338535) q[12];
rz(-5.543262115637049e-05) q[12];
ry(1.5708475176364571) q[13];
rz(-0.0003328831090172175) q[13];
ry(1.5568041747148047) q[14];
rz(-1.348159606517675) q[14];
ry(-0.00011080359010584574) q[15];
rz(-1.5512458207658848) q[15];
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
ry(1.5681727843530817) q[0];
rz(0.07801981130557946) q[0];
ry(1.5766172288227627) q[1];
rz(-1.6265314986999584) q[1];
ry(-1.6174332693837132) q[2];
rz(-2.0244057938446494) q[2];
ry(-1.5703942097637222) q[3];
rz(-6.850544017211746e-05) q[3];
ry(3.4887419785581506e-05) q[4];
rz(-1.8656740412006583) q[4];
ry(0.00011112397279902912) q[5];
rz(0.6473594473500897) q[5];
ry(-2.0070739659476073) q[6];
rz(-3.097799310929715) q[6];
ry(-1.5683822892649626) q[7];
rz(-1.5709437645597797) q[7];
ry(1.5705928783250387) q[8];
rz(-3.1371264249725117) q[8];
ry(-1.5707591442598758) q[9];
rz(-3.1232011131887307) q[9];
ry(0.00034896954869889635) q[10];
rz(0.19430380458532653) q[10];
ry(3.1403262314402323) q[11];
rz(2.0009617684090797) q[11];
ry(1.5710150272641838) q[12];
rz(-3.1401030207132976) q[12];
ry(-1.5707057928675017) q[13];
rz(1.7408669744745775) q[13];
ry(1.6525497201347426) q[14];
rz(0.00011398885449409589) q[14];
ry(-1.5714021521201404) q[15];
rz(1.6668194729512615) q[15];
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
ry(0.00015239499625222138) q[0];
rz(1.2916217914987698) q[0];
ry(3.1381155529718447) q[1];
rz(-0.9661590235055924) q[1];
ry(-0.008109147039975763) q[2];
rz(-1.3181580538859206) q[2];
ry(-1.572733880442927) q[3];
rz(2.2327158042573525) q[3];
ry(4.684167752328053e-05) q[4];
rz(2.2955117142226165) q[4];
ry(-1.5707748034851672) q[5];
rz(0.6620776749125833) q[5];
ry(-3.1398823482286247) q[6];
rz(1.406085850028588) q[6];
ry(1.57367833285792) q[7];
rz(2.2279117321287405) q[7];
ry(-1.5705742735509698) q[8];
rz(-0.20403742474238126) q[8];
ry(-3.137216800619705) q[9];
rz(-0.8896702833941948) q[9];
ry(-3.141536452602094) q[10];
rz(-1.5390272998012289) q[10];
ry(1.5707151506024521) q[11];
rz(2.232765749061545) q[11];
ry(-1.5151094770541143) q[12];
rz(-1.7770479399131383) q[12];
ry(0.00019821664071173473) q[13];
rz(-1.0810395770891021) q[13];
ry(1.5730249453959932) q[14];
rz(1.364865802979613) q[14];
ry(-0.002409398518200767) q[15];
rz(2.1374762486287135) q[15];