OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(0.06677398976025106) q[0];
rz(-2.531525119300594) q[0];
ry(2.2255305776941405) q[1];
rz(-1.9582039813592456) q[1];
ry(2.6321089078956046) q[2];
rz(-2.4006561628063747) q[2];
ry(2.607934175593508) q[3];
rz(-1.1052425631245077) q[3];
ry(3.141467435339346) q[4];
rz(1.446398382319582) q[4];
ry(-3.1411797919149635) q[5];
rz(-2.507368123177303) q[5];
ry(1.5212979254117114) q[6];
rz(-0.179195843643418) q[6];
ry(1.5917852797335168) q[7];
rz(2.1079005797320622) q[7];
ry(-3.1412017524381604) q[8];
rz(2.1454502330300644) q[8];
ry(3.141370900311917) q[9];
rz(-1.1959251567529026) q[9];
ry(1.5946244472360103) q[10];
rz(0.6971703859134953) q[10];
ry(-0.20355254933771763) q[11];
rz(2.6116260916642426) q[11];
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
ry(0.25559341361662175) q[0];
rz(-2.959920851438942) q[0];
ry(1.3975993575714938) q[1];
rz(-1.4702121022092847) q[1];
ry(-1.644686728796544) q[2];
rz(-2.7495573145934458) q[2];
ry(2.926535654658555) q[3];
rz(0.5834800996429415) q[3];
ry(-1.388724246783446) q[4];
rz(1.7604901613934225) q[4];
ry(6.625989263842058e-05) q[5];
rz(1.4976096311145186) q[5];
ry(0.8361468023536816) q[6];
rz(-0.9307805899511606) q[6];
ry(-1.788403530814605) q[7];
rz(-1.8047078149051914) q[7];
ry(3.1410701528280125) q[8];
rz(-0.9496412127999985) q[8];
ry(-1.2005750449131942e-05) q[9];
rz(-0.4996362208303821) q[9];
ry(2.6402521845803752) q[10];
rz(-2.2400377826800617) q[10];
ry(-1.61711675447421) q[11];
rz(-2.170345825378076) q[11];
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
ry(-2.0372030992768186) q[0];
rz(-2.902137162042355) q[0];
ry(-0.4476514429555424) q[1];
rz(2.960223228309216) q[1];
ry(3.045930181043241) q[2];
rz(-2.473862936280367) q[2];
ry(0.1458703200714023) q[3];
rz(-1.7624467981398284) q[3];
ry(3.1415369994566213) q[4];
rz(-0.12997851612317124) q[4];
ry(-0.7522934412532342) q[5];
rz(-1.8120096256344107) q[5];
ry(-0.0008836224282648348) q[6];
rz(1.223313913958727) q[6];
ry(-1.976410332971364) q[7];
rz(0.3703331904282905) q[7];
ry(-2.2733853216626927) q[8];
rz(0.5314446531506674) q[8];
ry(-3.141494215851173) q[9];
rz(2.312822410064328) q[9];
ry(-2.872560723183437) q[10];
rz(0.3707680280917223) q[10];
ry(-2.799865818335842) q[11];
rz(0.7969606115092329) q[11];
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
ry(0.18132515524146484) q[0];
rz(-1.9860270019310304) q[0];
ry(2.7556161255034484) q[1];
rz(-3.093614951421159) q[1];
ry(-0.5057928787642445) q[2];
rz(-0.023586972259413308) q[2];
ry(-3.1377885999333404) q[3];
rz(-1.0994186516663516) q[3];
ry(0.04657594451401892) q[4];
rz(-3.1058893833004273) q[4];
ry(3.1415493234800302) q[5];
rz(1.3285529627471844) q[5];
ry(-0.002630372418951385) q[6];
rz(-2.900559569564865) q[6];
ry(0.002515364968222154) q[7];
rz(-0.2315198249386451) q[7];
ry(3.140537048320028) q[8];
rz(1.2210010715113742) q[8];
ry(2.226997235339084e-05) q[9];
rz(1.1305650215339387) q[9];
ry(0.15216284850143058) q[10];
rz(-2.7542976268468062) q[10];
ry(1.3474011770770256) q[11];
rz(-0.8745408933149738) q[11];
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
ry(2.489048878773898) q[0];
rz(0.6927764209833928) q[0];
ry(-2.1723684095550504) q[1];
rz(-0.8151855003150771) q[1];
ry(-3.132768179403282) q[2];
rz(-0.9406617193710005) q[2];
ry(2.9149549354935997) q[3];
rz(-0.10878060225421421) q[3];
ry(3.141579060382847) q[4];
rz(3.0410647220815554) q[4];
ry(-2.3573683448539704) q[5];
rz(-1.1649813637115933) q[5];
ry(-3.1255810434657) q[6];
rz(-0.13405896491161418) q[6];
ry(-0.29579296737792493) q[7];
rz(-0.9863992733368064) q[7];
ry(0.8632814387636527) q[8];
rz(1.0585828652591296) q[8];
ry(0.00011707510065328859) q[9];
rz(-2.170802606476297) q[9];
ry(2.6735815106954286) q[10];
rz(1.5714930390167128) q[10];
ry(-2.935060854407883) q[11];
rz(1.8690334883269777) q[11];
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
ry(1.3806415134682064) q[0];
rz(1.1701418215551351) q[0];
ry(1.6744741688194473) q[1];
rz(0.46843197589449564) q[1];
ry(3.1372657266099075) q[2];
rz(-2.190801323338211) q[2];
ry(-0.08918014125870903) q[3];
rz(0.6559161994807879) q[3];
ry(0.6216213115721496) q[4];
rz(-3.1123980269581564) q[4];
ry(-3.141527383242826) q[5];
rz(1.3281197940552791) q[5];
ry(1.303161057626162) q[6];
rz(-0.002591881821213171) q[6];
ry(-1.603222075156591) q[7];
rz(1.5708800545079038) q[7];
ry(3.1390913520463104) q[8];
rz(2.0486965435498687) q[8];
ry(3.141533325277642) q[9];
rz(-1.29436019924304) q[9];
ry(-2.8239430494124447) q[10];
rz(0.5405819194469027) q[10];
ry(-2.090390537500652) q[11];
rz(-1.2511842143305925) q[11];
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
ry(0.9235278995498123) q[0];
rz(0.5182091019176651) q[0];
ry(-1.2027143183725053) q[1];
rz(2.192650773261274) q[1];
ry(2.7728934985446427) q[2];
rz(-0.1115745626113585) q[2];
ry(-1.4236619901208254) q[3];
rz(1.8715793773104652) q[3];
ry(5.532402438390704e-07) q[4];
rz(0.08149827121507336) q[4];
ry(3.140674001592701) q[5];
rz(-2.5232005493300242) q[5];
ry(0.0023836997231642103) q[6];
rz(0.002951827566329771) q[6];
ry(1.5484369887928433) q[7];
rz(-2.2533106335129087) q[7];
ry(-3.141380267117634) q[8];
rz(1.4143759488386192) q[8];
ry(-3.1414717220107025) q[9];
rz(-1.2582762332301685) q[9];
ry(0.5812297065592764) q[10];
rz(2.6282093020922255) q[10];
ry(0.3353895669488707) q[11];
rz(-0.9809343432254259) q[11];
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
ry(-0.4500896102835921) q[0];
rz(2.880561932596052) q[0];
ry(-1.5407669370048964) q[1];
rz(2.2369106133216867) q[1];
ry(-1.3718308289660346) q[2];
rz(-0.5518404295649569) q[2];
ry(-0.08271411703693644) q[3];
rz(-1.0927963167145576) q[3];
ry(2.941625876163096) q[4];
rz(0.004653215683913636) q[4];
ry(-4.215445297316921e-05) q[5];
rz(2.083868405266512) q[5];
ry(1.5353586413793123) q[6];
rz(0.505876917630605) q[6];
ry(0.12251256772540131) q[7];
rz(0.38854972422650247) q[7];
ry(-0.9149867809739645) q[8];
rz(-1.336466829536473) q[8];
ry(-3.1414902179778914) q[9];
rz(3.066380980184085) q[9];
ry(1.1876901763354786) q[10];
rz(0.6868081547413825) q[10];
ry(1.0367595154499551) q[11];
rz(-2.3829000225309804) q[11];
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
ry(-1.3359153981360352) q[0];
rz(-1.3001778523772307) q[0];
ry(-0.24858914506804197) q[1];
rz(-1.4678967273380588) q[1];
ry(1.3430954193790983) q[2];
rz(0.04019304575499149) q[2];
ry(-2.306550840923826) q[3];
rz(-1.2096683136535678) q[3];
ry(-0.00024560786506011567) q[4];
rz(3.0052039828262846) q[4];
ry(-0.0021466914735315257) q[5];
rz(-2.4260390577861966) q[5];
ry(-3.1415815074485214) q[6];
rz(-1.032198076004942) q[6];
ry(-1.0861959874428935) q[7];
rz(-2.5320871780750838) q[7];
ry(3.1411088227905952) q[8];
rz(2.0680633273850324) q[8];
ry(-3.1415262987796058) q[9];
rz(-0.01779344916621017) q[9];
ry(-3.1414076852410253) q[10];
rz(1.352495439737986) q[10];
ry(-2.3819652373049838) q[11];
rz(1.5598011525604747) q[11];
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
ry(-1.5811525792970147) q[0];
rz(0.8173978265352044) q[0];
ry(-2.409069557698242) q[1];
rz(-1.0836796161978695) q[1];
ry(-2.7336801944374565) q[2];
rz(-1.4263443418208412) q[2];
ry(1.3601312758423934) q[3];
rz(-0.7917838569804444) q[3];
ry(-0.31131691827446223) q[4];
rz(0.17704211573635928) q[4];
ry(0.00011534446051967418) q[5];
rz(-0.5652073221093739) q[5];
ry(-0.13244793628717666) q[6];
rz(2.2218621668742915) q[6];
ry(0.9234485961474472) q[7];
rz(0.9685608424223807) q[7];
ry(-1.3571102679018754) q[8];
rz(-2.912495853719934) q[8];
ry(0.00047466647603307033) q[9];
rz(-1.612588412621153) q[9];
ry(-3.0786185754623174) q[10];
rz(-0.5331794465190294) q[10];
ry(-1.383783548577129) q[11];
rz(-0.11489320463506691) q[11];
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
ry(2.0623846731683373) q[0];
rz(-1.0333700087588529) q[0];
ry(1.343027990890756) q[1];
rz(-1.4043960722299034) q[1];
ry(1.555368667478447) q[2];
rz(3.0113603612919055) q[2];
ry(-0.3788398455554025) q[3];
rz(-2.6205866141481646) q[3];
ry(0.0002528747587531263) q[4];
rz(-2.0961833471028952) q[4];
ry(3.1414032782006722) q[5];
rz(-2.7479966043445128) q[5];
ry(-3.1415678260922295) q[6];
rz(-1.7315697173208084) q[6];
ry(-0.12841349975251948) q[7];
rz(-1.5210542098296465) q[7];
ry(-0.0009723886461753771) q[8];
rz(-2.2256106827536812) q[8];
ry(3.14135401151391) q[9];
rz(-1.2742959984627742) q[9];
ry(-3.1397860674404767) q[10];
rz(1.231123454432586) q[10];
ry(-1.8393280863193047) q[11];
rz(-2.8334224590586885) q[11];
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
ry(1.6764904736985322) q[0];
rz(0.1754973361012144) q[0];
ry(-0.088870164083451) q[1];
rz(-0.14612298482199543) q[1];
ry(0.06291561710853344) q[2];
rz(1.4144478894707586) q[2];
ry(2.9070568458470825) q[3];
rz(-0.20917786945787512) q[3];
ry(-2.04983350181886) q[4];
rz(0.619297258961927) q[4];
ry(5.781228472390154e-05) q[5];
rz(-1.8620434902297993) q[5];
ry(-0.23827190230172215) q[6];
rz(1.957240898778629) q[6];
ry(-1.35007685203154) q[7];
rz(2.87583256177699) q[7];
ry(-2.396517539153327) q[8];
rz(0.8309664550777985) q[8];
ry(3.14142530278934) q[9];
rz(-1.060398467995742) q[9];
ry(-2.772039553449828) q[10];
rz(-2.625870520825772) q[10];
ry(0.6020283972711565) q[11];
rz(-1.5097992569059868) q[11];
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
ry(-0.17186851067369613) q[0];
rz(2.982670061669794) q[0];
ry(2.217542497701973) q[1];
rz(1.7747589764994487) q[1];
ry(0.09550332950982288) q[2];
rz(2.8642284769580892) q[2];
ry(-1.631596866572288) q[3];
rz(2.2539725580332988) q[3];
ry(3.1415701694973737) q[4];
rz(2.4966210131043933) q[4];
ry(-7.028659645491947e-05) q[5];
rz(1.8618447050015976) q[5];
ry(-0.0003670955518639829) q[6];
rz(-2.555554505061967) q[6];
ry(-1.5499553948919975) q[7];
rz(3.045383842303131) q[7];
ry(-3.1160121845237088) q[8];
rz(3.1036740120170547) q[8];
ry(-1.775163959925959) q[9];
rz(0.26342807493674325) q[9];
ry(-3.13895146924543) q[10];
rz(1.5582907977586204) q[10];
ry(1.0382683408550424) q[11];
rz(0.35776112956691186) q[11];
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
ry(1.4295982134278225) q[0];
rz(0.5347507886591258) q[0];
ry(-0.028546828792084908) q[1];
rz(0.2168109205039217) q[1];
ry(-3.048498155012682) q[2];
rz(2.541982654696985) q[2];
ry(-2.979111579484446) q[3];
rz(2.3527397861217234) q[3];
ry(-1.1443064130193474) q[4];
rz(-2.6532570915369877) q[4];
ry(0.28416035660615363) q[5];
rz(2.82172982736672) q[5];
ry(3.1332522257985898) q[6];
rz(0.6731791398795115) q[6];
ry(-0.0006242984009485752) q[7];
rz(2.882650895776285) q[7];
ry(-3.141219764746465) q[8];
rz(0.010894590137152882) q[8];
ry(0.000683308009448602) q[9];
rz(2.8782231032547667) q[9];
ry(0.07344361977967999) q[10];
rz(-0.6561136363650261) q[10];
ry(-3.1414233470059627) q[11];
rz(3.1186859032267398) q[11];
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
ry(-0.5960391264176081) q[0];
rz(-1.9192103859630034) q[0];
ry(0.2385281158800894) q[1];
rz(1.555467857775896) q[1];
ry(-1.5859286937015407) q[2];
rz(-1.5748269330128732) q[2];
ry(-0.00040322382592794526) q[3];
rz(-1.015381773406828) q[3];
ry(3.411270567325517e-05) q[4];
rz(-2.3299445758057553) q[4];
ry(3.9563368546602355e-05) q[5];
rz(0.23643838360384714) q[5];
ry(0.00017801096299052688) q[6];
rz(-0.5336370606453462) q[6];
ry(-1.5644731670207015) q[7];
rz(-1.705112693462471) q[7];
ry(3.115718964114167) q[8];
rz(2.054632758796614) q[8];
ry(1.3665268582972778) q[9];
rz(-1.2161406830839452) q[9];
ry(-3.1385311619419407) q[10];
rz(2.6168767060115434) q[10];
ry(1.2946350492270116) q[11];
rz(2.094135980746591) q[11];
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
ry(-1.1391893421545225) q[0];
rz(-0.20945470502463748) q[0];
ry(0.1958550894216824) q[1];
rz(1.2360815425261498) q[1];
ry(1.538907322741034) q[2];
rz(-0.08723939984364253) q[2];
ry(-1.1831997249015718) q[3];
rz(-1.0081962842762682) q[3];
ry(-1.407875652226773) q[4];
rz(2.1285251194696873) q[4];
ry(0.0020310105046374653) q[5];
rz(1.644071054199082) q[5];
ry(-1.739144313423532) q[6];
rz(-1.4565277981692462) q[6];
ry(1.4696427777205692) q[7];
rz(3.100706510418394) q[7];
ry(2.1416630879662573) q[8];
rz(2.213703755439687) q[8];
ry(-1.5705783250603478) q[9];
rz(1.200027346508324) q[9];
ry(1.85127134898367) q[10];
rz(0.7677755744686854) q[10];
ry(2.9583454507966898) q[11];
rz(-2.6223403420219706) q[11];
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
ry(0.15190459355327304) q[0];
rz(-2.0607065602921177) q[0];
ry(1.5602192925300713) q[1];
rz(2.837700857913224) q[1];
ry(-1.5674451354760504) q[2];
rz(1.385872780334208) q[2];
ry(-0.00016296623972684166) q[3];
rz(2.477890410361461) q[3];
ry(3.1415839383956787) q[4];
rz(-0.9492753072408711) q[4];
ry(1.5697386429828315) q[5];
rz(3.141505396203353) q[5];
ry(-3.1415638330613445) q[6];
rz(-1.5892102604473817) q[6];
ry(1.5707726748598911) q[7];
rz(0.9286875873657477) q[7];
ry(0.00016468292279685117) q[8];
rz(-0.05595066923771261) q[8];
ry(-0.04930905236125878) q[9];
rz(0.8635846543186051) q[9];
ry(-0.5444293473237937) q[10];
rz(-0.30495898001603017) q[10];
ry(1.5846905107591107) q[11];
rz(2.0908309661337103) q[11];
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
ry(-1.7665724211460203) q[0];
rz(1.5306732797882738) q[0];
ry(-2.9207765813734556) q[1];
rz(-2.5007148682207214) q[1];
ry(-2.938387252091318) q[2];
rz(-2.9395943858498903) q[2];
ry(0.049504985155887304) q[3];
rz(0.6382721262726587) q[3];
ry(-1.570771569957202) q[4];
rz(-1.1228208735700949) q[4];
ry(-1.5707934173493936) q[5];
rz(-1.5708019540541496) q[5];
ry(1.5748241225469695) q[6];
rz(0.8799776103149358) q[6];
ry(-3.1415739980599025) q[7];
rz(2.4883938091549944) q[7];
ry(-0.0002251919570406097) q[8];
rz(2.0930247304146015) q[8];
ry(-3.141481862267054) q[9];
rz(-1.9344777017858057) q[9];
ry(-0.8269039491565549) q[10];
rz(-0.2083148700156059) q[10];
ry(1.7134324809241948) q[11];
rz(-0.4342884074554677) q[11];
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
ry(-2.752235465991805) q[0];
rz(-0.9385214351382231) q[0];
ry(-0.002866655924727013) q[1];
rz(1.1268921922704727) q[1];
ry(0.00011843974159510921) q[2];
rz(2.420240478994749) q[2];
ry(-3.660059365930709e-05) q[3];
rz(0.9062928745476605) q[3];
ry(3.141499416245589) q[4];
rz(0.44661235233302654) q[4];
ry(1.570795947731556) q[5];
rz(1.4187399079956722) q[5];
ry(4.2033385856399264e-05) q[6];
rz(0.2020227996258049) q[6];
ry(1.2221827209983849) q[7];
rz(0.7673488534073396) q[7];
ry(9.384227348008852e-05) q[8];
rz(2.9069953268097826) q[8];
ry(-3.0048951143964326) q[9];
rz(-2.9439795307816548) q[9];
ry(-2.573381074682576) q[10];
rz(-3.1277537331925545) q[10];
ry(0.00600878615817605) q[11];
rz(1.6016157533939834) q[11];
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
ry(-2.155602543739089) q[0];
rz(1.1068242881745929) q[0];
ry(0.4929935526616101) q[1];
rz(-2.0847829690721857) q[1];
ry(-1.5666861259888565) q[2];
rz(0.20072331523672163) q[2];
ry(-1.5708002921818) q[3];
rz(1.4667035958636099) q[3];
ry(-1.5707966993837585) q[4];
rz(-1.1332388165356773) q[4];
ry(-3.141449394111068) q[5];
rz(3.0272729659681343) q[5];
ry(-0.1732531457268749) q[6];
rz(3.0401189197485188) q[6];
ry(-3.141068795771512) q[7];
rz(2.3620836488856547) q[7];
ry(3.1414384909097888) q[8];
rz(1.4717746365256286) q[8];
ry(-3.1415584326391284) q[9];
rz(-0.4364481355326263) q[9];
ry(-2.3633193591919706) q[10];
rz(-0.737805317448905) q[10];
ry(-1.617977120492035) q[11];
rz(-0.9577445539935657) q[11];
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
ry(1.625145284700107) q[0];
rz(-0.6270506691488525) q[0];
ry(-1.5726560742316276) q[1];
rz(1.2243944378338414) q[1];
ry(2.571183837852357) q[2];
rz(1.8981846544918761) q[2];
ry(-1.2813463465790444) q[3];
rz(-1.2668869296802665) q[3];
ry(-0.0003596842146098729) q[4];
rz(-1.0186232993469604) q[4];
ry(1.5284620689341235) q[5];
rz(1.4169417281482295) q[5];
ry(-3.1066456148310237) q[6];
rz(-1.098469717075547) q[6];
ry(-0.04549661154173812) q[7];
rz(-0.033445488678792726) q[7];
ry(-3.1415382266332017) q[8];
rz(-0.9432428173117929) q[8];
ry(-0.5801262891549612) q[9];
rz(-2.4973418136695424) q[9];
ry(0.14107432101426176) q[10];
rz(-2.5308254309219502) q[10];
ry(2.0167889859455603) q[11];
rz(-0.20680809586862206) q[11];
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
ry(1.5728771356050466) q[0];
rz(-3.141380773216549) q[0];
ry(3.0827682861870924) q[1];
rz(1.2238960677533612) q[1];
ry(-0.00010963610417036307) q[2];
rz(2.778723436172989) q[2];
ry(1.1677270447361427e-05) q[3];
rz(-1.9074134585524254) q[3];
ry(3.1415085800201936) q[4];
rz(0.9647541852485588) q[4];
ry(-1.2204707668495017e-05) q[5];
rz(-3.0275566691405866) q[5];
ry(2.974514286452367) q[6];
rz(-0.6493318134727525) q[6];
ry(-0.008937776806052837) q[7];
rz(1.549242384072239) q[7];
ry(1.8228032638477088e-05) q[8];
rz(-2.632903253305572) q[8];
ry(3.745544865285666e-05) q[9];
rz(2.872799039038881) q[9];
ry(0.034804713563591244) q[10];
rz(-1.03394450283583) q[10];
ry(-2.9862719346656013) q[11];
rz(1.0725481973828508) q[11];
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
ry(1.0034269749808487) q[0];
rz(-1.344123266666686) q[0];
ry(-0.5368532933762822) q[1];
rz(-1.3435903701017278) q[1];
ry(1.5489759189806875) q[2];
rz(-2.978104890986287) q[2];
ry(0.8482743126966723) q[3];
rz(-1.2546865273689694) q[3];
ry(-1.2529107060540237) q[4];
rz(1.742676834621646) q[4];
ry(1.2361690518676829) q[5];
rz(-1.3539350250562694) q[5];
ry(-2.8019523545211773) q[6];
rz(1.6056007575536209) q[6];
ry(1.631458813248067) q[7];
rz(-1.44512125934218) q[7];
ry(-1.5960541499911396) q[8];
rz(-2.97931918861587) q[8];
ry(1.9629919998413001) q[9];
rz(1.28908322541912) q[9];
ry(-1.0145790578430045) q[10];
rz(2.143818726221947) q[10];
ry(1.5438808668283306) q[11];
rz(-0.6983367657475634) q[11];