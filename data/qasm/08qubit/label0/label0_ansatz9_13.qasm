OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(0.2810667385001648) q[0];
ry(-1.169851129135954) q[1];
cx q[0],q[1];
ry(-0.25677382794801035) q[0];
ry(-2.8170864777235423) q[1];
cx q[0],q[1];
ry(2.699469056518411) q[2];
ry(2.370351012712744) q[3];
cx q[2],q[3];
ry(0.16362645631311779) q[2];
ry(-1.2399655221704275) q[3];
cx q[2],q[3];
ry(-2.024907107534718) q[4];
ry(2.681692270885869) q[5];
cx q[4],q[5];
ry(0.5569700096271015) q[4];
ry(-1.5816339907997214) q[5];
cx q[4],q[5];
ry(-0.06139313273196301) q[6];
ry(0.24476549601365277) q[7];
cx q[6],q[7];
ry(-1.8157416049084065) q[6];
ry(2.0592914829970255) q[7];
cx q[6],q[7];
ry(2.2694684865282255) q[0];
ry(1.9138200412162076) q[2];
cx q[0],q[2];
ry(0.3361849565846891) q[0];
ry(-1.4081920408508566) q[2];
cx q[0],q[2];
ry(-0.9220169927252275) q[2];
ry(-0.19652870691170032) q[4];
cx q[2],q[4];
ry(2.3367392173001855) q[2];
ry(-2.6746307964111957) q[4];
cx q[2],q[4];
ry(-1.6730292278455479) q[4];
ry(-2.3981708314081684) q[6];
cx q[4],q[6];
ry(-1.7021698130398615) q[4];
ry(2.1052796480974765) q[6];
cx q[4],q[6];
ry(0.8008553684829457) q[1];
ry(2.159731013998739) q[3];
cx q[1],q[3];
ry(2.0893557835808076) q[1];
ry(0.3269779633195276) q[3];
cx q[1],q[3];
ry(-2.854676282448236) q[3];
ry(-2.187006874951293) q[5];
cx q[3],q[5];
ry(2.7761954354461174) q[3];
ry(-1.128159080377685) q[5];
cx q[3],q[5];
ry(-1.5902096993058303) q[5];
ry(-1.820754031000245) q[7];
cx q[5],q[7];
ry(-0.8565264015457094) q[5];
ry(2.189868322545297) q[7];
cx q[5],q[7];
ry(-0.10446428101296143) q[0];
ry(-1.7333923169080065) q[3];
cx q[0],q[3];
ry(-2.531641882002683) q[0];
ry(-2.4592092558036995) q[3];
cx q[0],q[3];
ry(-3.085589867682932) q[1];
ry(-2.0060664111662883) q[2];
cx q[1],q[2];
ry(-1.597314966482183) q[1];
ry(1.0247193091949744) q[2];
cx q[1],q[2];
ry(3.054337392900803) q[2];
ry(-0.028009044422296903) q[5];
cx q[2],q[5];
ry(2.001984028639493) q[2];
ry(1.9886785398811453) q[5];
cx q[2],q[5];
ry(0.07860741081104675) q[3];
ry(-3.1072512950144646) q[4];
cx q[3],q[4];
ry(1.4244930263263003) q[3];
ry(1.0492727525069334) q[4];
cx q[3],q[4];
ry(-0.6763787133947069) q[4];
ry(-0.013715169634436458) q[7];
cx q[4],q[7];
ry(-3.0950751129288827) q[4];
ry(1.6504542258557509) q[7];
cx q[4],q[7];
ry(3.1036816602494803) q[5];
ry(0.29995676821725326) q[6];
cx q[5],q[6];
ry(-0.7259542884616978) q[5];
ry(-3.0945387453246744) q[6];
cx q[5],q[6];
ry(-1.2264799074713801) q[0];
ry(-1.5696096497165106) q[1];
cx q[0],q[1];
ry(1.3513983229289677) q[0];
ry(1.2221158821269724) q[1];
cx q[0],q[1];
ry(-2.6809994698545974) q[2];
ry(-2.162888821722964) q[3];
cx q[2],q[3];
ry(-0.6838262040866114) q[2];
ry(1.495182358167684) q[3];
cx q[2],q[3];
ry(-2.992614491371667) q[4];
ry(2.88755883685891) q[5];
cx q[4],q[5];
ry(2.3217209079718057) q[4];
ry(-2.523082620220367) q[5];
cx q[4],q[5];
ry(-2.876811892995229) q[6];
ry(-1.9170820932483847) q[7];
cx q[6],q[7];
ry(2.7913265415672557) q[6];
ry(-2.8060523430146604) q[7];
cx q[6],q[7];
ry(-0.13836861703305647) q[0];
ry(-1.0823123543142756) q[2];
cx q[0],q[2];
ry(-2.893052431199517) q[0];
ry(2.0566591921422934) q[2];
cx q[0],q[2];
ry(1.674594197436552) q[2];
ry(-0.7543114350787805) q[4];
cx q[2],q[4];
ry(-0.4672890653597426) q[2];
ry(2.71172761618035) q[4];
cx q[2],q[4];
ry(-2.293703681182291) q[4];
ry(-1.5221602907146703) q[6];
cx q[4],q[6];
ry(0.1609663247565074) q[4];
ry(2.986385292729389) q[6];
cx q[4],q[6];
ry(1.5460379121699732) q[1];
ry(0.8107171018881206) q[3];
cx q[1],q[3];
ry(2.7529135912481015) q[1];
ry(2.237928843293981) q[3];
cx q[1],q[3];
ry(2.043481166481615) q[3];
ry(1.1442606461144025) q[5];
cx q[3],q[5];
ry(0.9593725376966891) q[3];
ry(-0.5925907512809303) q[5];
cx q[3],q[5];
ry(-1.1297378884675178) q[5];
ry(-0.10049549581337291) q[7];
cx q[5],q[7];
ry(0.7850540146597768) q[5];
ry(2.4311146780480364) q[7];
cx q[5],q[7];
ry(1.4259614656246369) q[0];
ry(-2.00464870907592) q[3];
cx q[0],q[3];
ry(0.62500269383529) q[0];
ry(-0.7776761190224706) q[3];
cx q[0],q[3];
ry(-0.40562849177390886) q[1];
ry(1.5880847513521354) q[2];
cx q[1],q[2];
ry(-0.17000364963885575) q[1];
ry(-2.610098094975987) q[2];
cx q[1],q[2];
ry(0.9637135874197407) q[2];
ry(-0.25809240662203414) q[5];
cx q[2],q[5];
ry(0.903447174667197) q[2];
ry(2.6032296782291366) q[5];
cx q[2],q[5];
ry(0.7122934897756483) q[3];
ry(-0.5427449013803632) q[4];
cx q[3],q[4];
ry(2.5567450089878765) q[3];
ry(2.349409399949228) q[4];
cx q[3],q[4];
ry(-1.0456055825212698) q[4];
ry(1.311226031775619) q[7];
cx q[4],q[7];
ry(1.7691527791331838) q[4];
ry(2.9917214788411957) q[7];
cx q[4],q[7];
ry(2.5044505215575743) q[5];
ry(0.8382419522848724) q[6];
cx q[5],q[6];
ry(2.5074357167663712) q[5];
ry(0.08297619703565662) q[6];
cx q[5],q[6];
ry(-0.05686945907453314) q[0];
ry(1.7230422543600596) q[1];
cx q[0],q[1];
ry(-0.45489910221139335) q[0];
ry(1.5061038294137727) q[1];
cx q[0],q[1];
ry(2.86159222596263) q[2];
ry(2.54922154998763) q[3];
cx q[2],q[3];
ry(-1.9865688495271296) q[2];
ry(2.2654054959905228) q[3];
cx q[2],q[3];
ry(-0.41663014405408566) q[4];
ry(-1.7997767579472452) q[5];
cx q[4],q[5];
ry(3.0416601316829013) q[4];
ry(2.2393415743136256) q[5];
cx q[4],q[5];
ry(-1.2776946908936975) q[6];
ry(-2.940415986280815) q[7];
cx q[6],q[7];
ry(-2.3773228435110463) q[6];
ry(1.5681070499589722) q[7];
cx q[6],q[7];
ry(-2.4036174145836466) q[0];
ry(1.0175244404478825) q[2];
cx q[0],q[2];
ry(-0.6113292085390296) q[0];
ry(-2.2474021678292764) q[2];
cx q[0],q[2];
ry(0.4941214989936237) q[2];
ry(-1.7930208137751449) q[4];
cx q[2],q[4];
ry(-1.5297866956313035) q[2];
ry(1.1752115679988089) q[4];
cx q[2],q[4];
ry(1.411853479241508) q[4];
ry(-2.662743021628108) q[6];
cx q[4],q[6];
ry(1.3870467172598) q[4];
ry(-0.36550228835389387) q[6];
cx q[4],q[6];
ry(0.4954816971559114) q[1];
ry(-2.933355452563982) q[3];
cx q[1],q[3];
ry(-1.0926065584421065) q[1];
ry(2.740020666240076) q[3];
cx q[1],q[3];
ry(-1.4403220981574676) q[3];
ry(2.089035471466505) q[5];
cx q[3],q[5];
ry(-1.4021979822494075) q[3];
ry(-2.108455284071134) q[5];
cx q[3],q[5];
ry(2.8239895865475564) q[5];
ry(2.7955483702029333) q[7];
cx q[5],q[7];
ry(0.3850184459876856) q[5];
ry(-2.4158161780255556) q[7];
cx q[5],q[7];
ry(-0.6214977627033207) q[0];
ry(-2.89124030360972) q[3];
cx q[0],q[3];
ry(-2.3757946121004054) q[0];
ry(-0.5028186297077628) q[3];
cx q[0],q[3];
ry(1.992162174240394) q[1];
ry(-1.1868640529388088) q[2];
cx q[1],q[2];
ry(0.7648008965375878) q[1];
ry(1.0251421855866862) q[2];
cx q[1],q[2];
ry(-1.2150820683299333) q[2];
ry(1.9785631247843751) q[5];
cx q[2],q[5];
ry(-2.483116474301306) q[2];
ry(1.923492991394924) q[5];
cx q[2],q[5];
ry(0.1287717569051665) q[3];
ry(2.3467702742672736) q[4];
cx q[3],q[4];
ry(-0.0010822806176218336) q[3];
ry(0.1301914142095928) q[4];
cx q[3],q[4];
ry(1.2905441873027585) q[4];
ry(-2.469736390976111) q[7];
cx q[4],q[7];
ry(0.8559935163088959) q[4];
ry(-1.1774533468905788) q[7];
cx q[4],q[7];
ry(-3.0892421020779697) q[5];
ry(-0.8463839421824728) q[6];
cx q[5],q[6];
ry(1.034735731612554) q[5];
ry(-1.8250608163829192) q[6];
cx q[5],q[6];
ry(-1.9040243235433785) q[0];
ry(-0.5791190551258951) q[1];
cx q[0],q[1];
ry(2.0732422862698248) q[0];
ry(-0.8905995804300786) q[1];
cx q[0],q[1];
ry(-1.1782990737990293) q[2];
ry(1.9887056066318103) q[3];
cx q[2],q[3];
ry(2.4533229741756983) q[2];
ry(-1.6709341700512308) q[3];
cx q[2],q[3];
ry(1.6818188633200972) q[4];
ry(-0.7424952176534071) q[5];
cx q[4],q[5];
ry(0.36738539791152386) q[4];
ry(-1.1465284273818979) q[5];
cx q[4],q[5];
ry(-1.210732088046847) q[6];
ry(1.8103779005062735) q[7];
cx q[6],q[7];
ry(1.1703348439042762) q[6];
ry(-0.8811901904021492) q[7];
cx q[6],q[7];
ry(-2.894898953967176) q[0];
ry(0.03540619594673711) q[2];
cx q[0],q[2];
ry(-1.9985985263700246) q[0];
ry(1.6109750237687095) q[2];
cx q[0],q[2];
ry(1.63962865060478) q[2];
ry(1.2971934912066323) q[4];
cx q[2],q[4];
ry(-1.5874616522147194) q[2];
ry(-2.5469615339476417) q[4];
cx q[2],q[4];
ry(0.5105097679022642) q[4];
ry(0.7804220483203826) q[6];
cx q[4],q[6];
ry(0.8317973594131383) q[4];
ry(2.7409593398901637) q[6];
cx q[4],q[6];
ry(-1.5042181091261861) q[1];
ry(-2.559206270826125) q[3];
cx q[1],q[3];
ry(0.5933450488235071) q[1];
ry(0.2799288174840666) q[3];
cx q[1],q[3];
ry(-1.4931813848558828) q[3];
ry(-1.7889119560041544) q[5];
cx q[3],q[5];
ry(-2.1100190044223233) q[3];
ry(1.400772461097199) q[5];
cx q[3],q[5];
ry(1.6129604173076895) q[5];
ry(-0.6344821786023314) q[7];
cx q[5],q[7];
ry(1.2261996144476492) q[5];
ry(2.3026246328427886) q[7];
cx q[5],q[7];
ry(-1.7115168779948906) q[0];
ry(1.77267705572977) q[3];
cx q[0],q[3];
ry(0.10977093671830485) q[0];
ry(1.2132311020844022) q[3];
cx q[0],q[3];
ry(1.6586729926833317) q[1];
ry(1.6219611005738335) q[2];
cx q[1],q[2];
ry(1.1499218776537827) q[1];
ry(0.8475657070382016) q[2];
cx q[1],q[2];
ry(2.5390585731147777) q[2];
ry(1.3051528437094495) q[5];
cx q[2],q[5];
ry(-0.7211064550178299) q[2];
ry(-1.9193589450556596) q[5];
cx q[2],q[5];
ry(0.8000746543172376) q[3];
ry(-1.2581934411087616) q[4];
cx q[3],q[4];
ry(1.7481630530580377) q[3];
ry(1.7608757905315064) q[4];
cx q[3],q[4];
ry(1.1925375282642263) q[4];
ry(2.341921622166426) q[7];
cx q[4],q[7];
ry(2.983361724820281) q[4];
ry(2.7558764869344543) q[7];
cx q[4],q[7];
ry(-0.8690482161112006) q[5];
ry(-2.588198834437428) q[6];
cx q[5],q[6];
ry(-0.1561892394195521) q[5];
ry(0.7175350074594684) q[6];
cx q[5],q[6];
ry(0.5880675391454016) q[0];
ry(1.6102945367416996) q[1];
cx q[0],q[1];
ry(1.004361902597947) q[0];
ry(-3.040363364869052) q[1];
cx q[0],q[1];
ry(0.9500418012900157) q[2];
ry(2.77356178623883) q[3];
cx q[2],q[3];
ry(-2.749890758914043) q[2];
ry(-2.4302416901601704) q[3];
cx q[2],q[3];
ry(-1.6710353446372337) q[4];
ry(-1.0807766322791423) q[5];
cx q[4],q[5];
ry(2.958496248663816) q[4];
ry(1.0275357974106623) q[5];
cx q[4],q[5];
ry(-1.614306297653304) q[6];
ry(-0.1342134482502111) q[7];
cx q[6],q[7];
ry(-2.7064570482315853) q[6];
ry(-1.2662146994999377) q[7];
cx q[6],q[7];
ry(-2.2738723815364055) q[0];
ry(2.325695957208423) q[2];
cx q[0],q[2];
ry(0.2079450254271249) q[0];
ry(-2.0852667973843904) q[2];
cx q[0],q[2];
ry(-1.3087500193302315) q[2];
ry(-1.3444655789387605) q[4];
cx q[2],q[4];
ry(0.7626983610531562) q[2];
ry(1.3217014547773653) q[4];
cx q[2],q[4];
ry(1.938611258759768) q[4];
ry(0.8733739482070435) q[6];
cx q[4],q[6];
ry(2.043857279861701) q[4];
ry(0.1110595456449408) q[6];
cx q[4],q[6];
ry(-0.935309611915728) q[1];
ry(-1.614175161465928) q[3];
cx q[1],q[3];
ry(1.7928158752853556) q[1];
ry(-0.019653349112125112) q[3];
cx q[1],q[3];
ry(-1.5824719985635776) q[3];
ry(-0.20908512505374777) q[5];
cx q[3],q[5];
ry(0.11220998395776771) q[3];
ry(-0.7048494907192784) q[5];
cx q[3],q[5];
ry(-2.5334743842181604) q[5];
ry(2.978902316636984) q[7];
cx q[5],q[7];
ry(3.060926238571976) q[5];
ry(1.6529632380473425) q[7];
cx q[5],q[7];
ry(2.0532112402773746) q[0];
ry(0.16623171419191696) q[3];
cx q[0],q[3];
ry(1.4962899264859422) q[0];
ry(2.994653891577211) q[3];
cx q[0],q[3];
ry(-2.000642534398465) q[1];
ry(-0.3301709290169913) q[2];
cx q[1],q[2];
ry(-1.6083765403143726) q[1];
ry(0.7788292276589418) q[2];
cx q[1],q[2];
ry(-0.2527582751097372) q[2];
ry(1.639707126847152) q[5];
cx q[2],q[5];
ry(3.1350616089512444) q[2];
ry(-1.6904352408072647) q[5];
cx q[2],q[5];
ry(-0.830616883113176) q[3];
ry(0.0021758666210107695) q[4];
cx q[3],q[4];
ry(2.169933340771764) q[3];
ry(1.1858431174688437) q[4];
cx q[3],q[4];
ry(-3.02801166707289) q[4];
ry(-0.39953879960552335) q[7];
cx q[4],q[7];
ry(-1.5892324513796607) q[4];
ry(-0.2341233246471326) q[7];
cx q[4],q[7];
ry(1.9188976896117793) q[5];
ry(3.0901960917256566) q[6];
cx q[5],q[6];
ry(-1.1907669706781783) q[5];
ry(-0.5183326572788519) q[6];
cx q[5],q[6];
ry(-2.091130823722022) q[0];
ry(-1.6034937484872414) q[1];
cx q[0],q[1];
ry(-1.5380067275996234) q[0];
ry(0.8984547586351742) q[1];
cx q[0],q[1];
ry(0.06716620987209687) q[2];
ry(-3.0545155274243037) q[3];
cx q[2],q[3];
ry(1.715341304295226) q[2];
ry(-1.8883264777257138) q[3];
cx q[2],q[3];
ry(2.8664339098352394) q[4];
ry(-0.6206304508576466) q[5];
cx q[4],q[5];
ry(-2.4204827440589227) q[4];
ry(-0.15450564447750992) q[5];
cx q[4],q[5];
ry(-2.2319284117326825) q[6];
ry(-1.1204681115433668) q[7];
cx q[6],q[7];
ry(1.5459180691477306) q[6];
ry(-2.311379676111555) q[7];
cx q[6],q[7];
ry(2.711528721014026) q[0];
ry(-1.386775784133813) q[2];
cx q[0],q[2];
ry(-0.7863068004937466) q[0];
ry(2.438593091798004) q[2];
cx q[0],q[2];
ry(1.365081277692192) q[2];
ry(-1.5552846461957746) q[4];
cx q[2],q[4];
ry(-0.9918360780038418) q[2];
ry(-2.2924169206849756) q[4];
cx q[2],q[4];
ry(0.5283318946044062) q[4];
ry(-0.5405040756493368) q[6];
cx q[4],q[6];
ry(-1.7313230638673076) q[4];
ry(-1.341864752632464) q[6];
cx q[4],q[6];
ry(0.38951642164229167) q[1];
ry(2.1092389242404326) q[3];
cx q[1],q[3];
ry(-1.0369105642960035) q[1];
ry(-2.1663816247659557) q[3];
cx q[1],q[3];
ry(-1.9118650014754284) q[3];
ry(-0.35906229988584604) q[5];
cx q[3],q[5];
ry(1.596875987816353) q[3];
ry(2.755127693872865) q[5];
cx q[3],q[5];
ry(-2.4335409181674303) q[5];
ry(0.9282348863304879) q[7];
cx q[5],q[7];
ry(0.8166429822331115) q[5];
ry(-1.3715120414570983) q[7];
cx q[5],q[7];
ry(-2.2878444660778277) q[0];
ry(-0.1326268581829466) q[3];
cx q[0],q[3];
ry(2.0866501656731895) q[0];
ry(-1.6095225800086945) q[3];
cx q[0],q[3];
ry(2.2197644258386355) q[1];
ry(-0.39460261568478305) q[2];
cx q[1],q[2];
ry(-0.9185583582127527) q[1];
ry(-1.8275599993378089) q[2];
cx q[1],q[2];
ry(-1.0803128632161778) q[2];
ry(1.3684520273801528) q[5];
cx q[2],q[5];
ry(1.7374441654404942) q[2];
ry(-0.3810113483855391) q[5];
cx q[2],q[5];
ry(-2.670017973477914) q[3];
ry(-0.28138039666732156) q[4];
cx q[3],q[4];
ry(-1.8702909217354144) q[3];
ry(1.7789210213058932) q[4];
cx q[3],q[4];
ry(-2.896539010292072) q[4];
ry(-1.5635362543508367) q[7];
cx q[4],q[7];
ry(-2.5850091910456086) q[4];
ry(-1.9820802046088906) q[7];
cx q[4],q[7];
ry(1.1395440736564204) q[5];
ry(-1.0896784282730623) q[6];
cx q[5],q[6];
ry(2.9279108686591617) q[5];
ry(-1.7885410137738873) q[6];
cx q[5],q[6];
ry(-1.4601556575200387) q[0];
ry(-1.263892200547665) q[1];
cx q[0],q[1];
ry(1.1691710510119648) q[0];
ry(0.0801110022426057) q[1];
cx q[0],q[1];
ry(1.4453193548249845) q[2];
ry(-2.2721318685877234) q[3];
cx q[2],q[3];
ry(0.0719175585098393) q[2];
ry(1.3411008292595252) q[3];
cx q[2],q[3];
ry(2.939003352098554) q[4];
ry(1.9201950630497642) q[5];
cx q[4],q[5];
ry(-0.46242709617073796) q[4];
ry(2.9119480278625143) q[5];
cx q[4],q[5];
ry(-1.79210789051338) q[6];
ry(1.2102857791372266) q[7];
cx q[6],q[7];
ry(-2.1124583643955943) q[6];
ry(0.3581479661318616) q[7];
cx q[6],q[7];
ry(-1.5924400991298517) q[0];
ry(-2.047243763547069) q[2];
cx q[0],q[2];
ry(-2.9758764625583964) q[0];
ry(-0.5020600860953468) q[2];
cx q[0],q[2];
ry(1.1758907559041933) q[2];
ry(-2.9338616239062523) q[4];
cx q[2],q[4];
ry(-0.9781937604458149) q[2];
ry(-2.7173565893036753) q[4];
cx q[2],q[4];
ry(2.371587153571195) q[4];
ry(-0.2511191041176886) q[6];
cx q[4],q[6];
ry(2.881795628766951) q[4];
ry(-2.5332218130743542) q[6];
cx q[4],q[6];
ry(-2.9118384036245546) q[1];
ry(-1.3138050247455482) q[3];
cx q[1],q[3];
ry(2.448429074474905) q[1];
ry(-2.52687038813694) q[3];
cx q[1],q[3];
ry(1.6847480982506733) q[3];
ry(2.196182379699719) q[5];
cx q[3],q[5];
ry(-0.48917933601276786) q[3];
ry(-0.29094535435815594) q[5];
cx q[3],q[5];
ry(3.057687781875232) q[5];
ry(2.702434168449181) q[7];
cx q[5],q[7];
ry(-1.8345028741279186) q[5];
ry(2.5918196450512556) q[7];
cx q[5],q[7];
ry(2.325260909540561) q[0];
ry(2.7753748156106255) q[3];
cx q[0],q[3];
ry(-2.3778645508785736) q[0];
ry(1.6966723640096484) q[3];
cx q[0],q[3];
ry(-1.7462736398704664) q[1];
ry(-2.1467472684446918) q[2];
cx q[1],q[2];
ry(-1.0705970026833755) q[1];
ry(-0.9426842149588712) q[2];
cx q[1],q[2];
ry(0.41894976694549513) q[2];
ry(-0.41953418610656) q[5];
cx q[2],q[5];
ry(0.9616731150617871) q[2];
ry(-0.25383508453243264) q[5];
cx q[2],q[5];
ry(-1.855355847461997) q[3];
ry(-1.304340581274159) q[4];
cx q[3],q[4];
ry(0.6939648267540868) q[3];
ry(2.8286193040142127) q[4];
cx q[3],q[4];
ry(0.03948220517432656) q[4];
ry(-2.112977878442175) q[7];
cx q[4],q[7];
ry(2.640920578412635) q[4];
ry(-2.0973853813487997) q[7];
cx q[4],q[7];
ry(2.0230649325507866) q[5];
ry(1.397634107432108) q[6];
cx q[5],q[6];
ry(0.7440716267910412) q[5];
ry(-1.9167600388312653) q[6];
cx q[5],q[6];
ry(0.4682483900412109) q[0];
ry(-2.243306772852445) q[1];
cx q[0],q[1];
ry(0.6120211441747735) q[0];
ry(1.1973911355619453) q[1];
cx q[0],q[1];
ry(-3.0404461887528997) q[2];
ry(1.3217636023007033) q[3];
cx q[2],q[3];
ry(1.6648531684660086) q[2];
ry(0.730279216576326) q[3];
cx q[2],q[3];
ry(-0.8130415792831732) q[4];
ry(2.63036493035964) q[5];
cx q[4],q[5];
ry(0.24986085679372108) q[4];
ry(3.067337023531792) q[5];
cx q[4],q[5];
ry(1.2934612545580944) q[6];
ry(2.7406438455449544) q[7];
cx q[6],q[7];
ry(-0.7363552821814281) q[6];
ry(2.803373199482184) q[7];
cx q[6],q[7];
ry(-0.057672835884499306) q[0];
ry(-0.8676188629735835) q[2];
cx q[0],q[2];
ry(2.007843146745286) q[0];
ry(-1.1885183365273706) q[2];
cx q[0],q[2];
ry(2.592232298725657) q[2];
ry(-2.226566140766275) q[4];
cx q[2],q[4];
ry(2.060274885571701) q[2];
ry(-0.12366146091073027) q[4];
cx q[2],q[4];
ry(-1.8349005497709836) q[4];
ry(-0.05466550494993771) q[6];
cx q[4],q[6];
ry(0.14605701588336117) q[4];
ry(-1.4352206039723456) q[6];
cx q[4],q[6];
ry(-1.8816242760929587) q[1];
ry(-1.4699209537669642) q[3];
cx q[1],q[3];
ry(2.526672838085367) q[1];
ry(0.1791523661187373) q[3];
cx q[1],q[3];
ry(-0.6251069036178051) q[3];
ry(-0.3340704783187313) q[5];
cx q[3],q[5];
ry(-2.34342301292713) q[3];
ry(0.2154642077255252) q[5];
cx q[3],q[5];
ry(1.33630858245753) q[5];
ry(0.5774529325916227) q[7];
cx q[5],q[7];
ry(-1.332149436704694) q[5];
ry(2.164827980981648) q[7];
cx q[5],q[7];
ry(-2.126696005522344) q[0];
ry(2.2108755255389734) q[3];
cx q[0],q[3];
ry(1.1917733587577717) q[0];
ry(1.773546004130206) q[3];
cx q[0],q[3];
ry(1.674028355153592) q[1];
ry(-1.973647847065618) q[2];
cx q[1],q[2];
ry(0.5513650121598417) q[1];
ry(1.138584614679373) q[2];
cx q[1],q[2];
ry(-3.082637104282245) q[2];
ry(0.4189050174255645) q[5];
cx q[2],q[5];
ry(-2.3467473889316226) q[2];
ry(0.36044350945772435) q[5];
cx q[2],q[5];
ry(-0.750242783257244) q[3];
ry(2.43332772342595) q[4];
cx q[3],q[4];
ry(-0.37787909486297) q[3];
ry(1.0431208373509886) q[4];
cx q[3],q[4];
ry(2.4906110521887244) q[4];
ry(-0.34574566749686303) q[7];
cx q[4],q[7];
ry(2.9086860641723313) q[4];
ry(2.412363164917687) q[7];
cx q[4],q[7];
ry(-0.9953863869577542) q[5];
ry(1.2551462961088102) q[6];
cx q[5],q[6];
ry(1.0501814017628917) q[5];
ry(-3.1293838687574422) q[6];
cx q[5],q[6];
ry(2.513182606973096) q[0];
ry(-1.4893471420903495) q[1];
cx q[0],q[1];
ry(-2.7385825921851716) q[0];
ry(1.3480034815677913) q[1];
cx q[0],q[1];
ry(-0.2891949966142535) q[2];
ry(-2.699868811628289) q[3];
cx q[2],q[3];
ry(1.3383609028676804) q[2];
ry(1.0569378279250354) q[3];
cx q[2],q[3];
ry(-0.6270052990025335) q[4];
ry(0.14540498934165402) q[5];
cx q[4],q[5];
ry(0.44600966442507506) q[4];
ry(2.604908851791913) q[5];
cx q[4],q[5];
ry(0.1281679206450148) q[6];
ry(2.3079900768845016) q[7];
cx q[6],q[7];
ry(0.20334972184346667) q[6];
ry(-1.5060715713273527) q[7];
cx q[6],q[7];
ry(-1.1523749570499158) q[0];
ry(0.37874846795140893) q[2];
cx q[0],q[2];
ry(1.6657935349432553) q[0];
ry(1.4370564374501926) q[2];
cx q[0],q[2];
ry(-0.39110867124949955) q[2];
ry(-2.727513890386635) q[4];
cx q[2],q[4];
ry(0.8150074730787095) q[2];
ry(0.09821334024314421) q[4];
cx q[2],q[4];
ry(2.703457022837574) q[4];
ry(-0.3194785630689487) q[6];
cx q[4],q[6];
ry(-1.1130112972262671) q[4];
ry(-0.5768793492653209) q[6];
cx q[4],q[6];
ry(-2.5417916400634386) q[1];
ry(0.8124637280053901) q[3];
cx q[1],q[3];
ry(2.0176960419393812) q[1];
ry(2.7001848210631993) q[3];
cx q[1],q[3];
ry(0.6754306352875208) q[3];
ry(-0.30844003793290914) q[5];
cx q[3],q[5];
ry(-1.0560561385180733) q[3];
ry(-2.415011826759444) q[5];
cx q[3],q[5];
ry(-0.9542130112397329) q[5];
ry(2.097483141057664) q[7];
cx q[5],q[7];
ry(2.3178329989160478) q[5];
ry(1.010825296251536) q[7];
cx q[5],q[7];
ry(-2.754892758393211) q[0];
ry(-1.454525342015077) q[3];
cx q[0],q[3];
ry(0.23649053722145386) q[0];
ry(-2.253750402604023) q[3];
cx q[0],q[3];
ry(-2.2107008537749184) q[1];
ry(-2.826840279339472) q[2];
cx q[1],q[2];
ry(2.2409817166803365) q[1];
ry(0.6775275970402977) q[2];
cx q[1],q[2];
ry(-3.071085500959538) q[2];
ry(2.554185713687431) q[5];
cx q[2],q[5];
ry(-2.7937291454709947) q[2];
ry(0.32265669933512076) q[5];
cx q[2],q[5];
ry(0.7549642876296886) q[3];
ry(-2.4396827549573095) q[4];
cx q[3],q[4];
ry(-1.4509395659104776) q[3];
ry(0.5796995842960939) q[4];
cx q[3],q[4];
ry(-1.4050380048988749) q[4];
ry(1.3923787845782218) q[7];
cx q[4],q[7];
ry(-2.7811382617681586) q[4];
ry(-1.5692698108947312) q[7];
cx q[4],q[7];
ry(-2.7715001751972466) q[5];
ry(-1.6881831578179038) q[6];
cx q[5],q[6];
ry(-2.64732142906349) q[5];
ry(0.31333826342041604) q[6];
cx q[5],q[6];
ry(-2.028302618017806) q[0];
ry(0.2274761832519454) q[1];
cx q[0],q[1];
ry(-1.274478679066224) q[0];
ry(2.5397411401092227) q[1];
cx q[0],q[1];
ry(-2.9748279902691905) q[2];
ry(2.9152738347373077) q[3];
cx q[2],q[3];
ry(-2.3335200289801756) q[2];
ry(-0.6265246426680016) q[3];
cx q[2],q[3];
ry(-1.62956010570033) q[4];
ry(0.3580747897179517) q[5];
cx q[4],q[5];
ry(-2.0132188744417236) q[4];
ry(-1.8529421645147703) q[5];
cx q[4],q[5];
ry(-2.5793660297923404) q[6];
ry(-1.5912496348257679) q[7];
cx q[6],q[7];
ry(1.8045483553214376) q[6];
ry(-0.4743428060341356) q[7];
cx q[6],q[7];
ry(0.5405778984140126) q[0];
ry(2.562204593177366) q[2];
cx q[0],q[2];
ry(-1.6362191664594061) q[0];
ry(-2.1169483214107636) q[2];
cx q[0],q[2];
ry(1.1595139771531286) q[2];
ry(0.5001712058870512) q[4];
cx q[2],q[4];
ry(1.123189904700694) q[2];
ry(-1.9292447316770236) q[4];
cx q[2],q[4];
ry(-1.4741545557229134) q[4];
ry(2.6068117986178905) q[6];
cx q[4],q[6];
ry(0.2585115600504557) q[4];
ry(1.1564334848261657) q[6];
cx q[4],q[6];
ry(-0.5512402376322002) q[1];
ry(2.838545190056274) q[3];
cx q[1],q[3];
ry(-1.1856037919752325) q[1];
ry(2.8000916587820397) q[3];
cx q[1],q[3];
ry(0.02389277428739156) q[3];
ry(-1.0636532037034794) q[5];
cx q[3],q[5];
ry(1.3126956142046504) q[3];
ry(-0.036499969502318284) q[5];
cx q[3],q[5];
ry(-0.07289033916546561) q[5];
ry(-1.1387638606383632) q[7];
cx q[5],q[7];
ry(0.7631308366375645) q[5];
ry(2.7966743440691806) q[7];
cx q[5],q[7];
ry(-0.7770887167684893) q[0];
ry(-2.206860641453345) q[3];
cx q[0],q[3];
ry(2.16779612516767) q[0];
ry(0.043531119423599246) q[3];
cx q[0],q[3];
ry(0.13137044372090165) q[1];
ry(-1.6735207289565663) q[2];
cx q[1],q[2];
ry(0.10578393511058648) q[1];
ry(-1.0328722980704192) q[2];
cx q[1],q[2];
ry(1.6679830386574195) q[2];
ry(-1.825120927179535) q[5];
cx q[2],q[5];
ry(1.7795853648079065) q[2];
ry(-0.36736217401207694) q[5];
cx q[2],q[5];
ry(-1.643281559340763) q[3];
ry(-0.3543123956780774) q[4];
cx q[3],q[4];
ry(1.9989952830746) q[3];
ry(1.3478053774153476) q[4];
cx q[3],q[4];
ry(2.0988536836182075) q[4];
ry(-0.8994720162961647) q[7];
cx q[4],q[7];
ry(0.39155788450423934) q[4];
ry(-0.4815121069707558) q[7];
cx q[4],q[7];
ry(1.5873984204699578) q[5];
ry(-0.8810421306964475) q[6];
cx q[5],q[6];
ry(1.1011469435450358) q[5];
ry(-0.36215258556835295) q[6];
cx q[5],q[6];
ry(0.9371519604084816) q[0];
ry(0.24007489562074472) q[1];
cx q[0],q[1];
ry(-1.5388182633062306) q[0];
ry(-2.3057345862698493) q[1];
cx q[0],q[1];
ry(0.5401896248398526) q[2];
ry(0.7522831824797358) q[3];
cx q[2],q[3];
ry(-2.353102109065977) q[2];
ry(1.1300235523217195) q[3];
cx q[2],q[3];
ry(-1.3448678525735565) q[4];
ry(-1.7809731505661461) q[5];
cx q[4],q[5];
ry(2.674458591383327) q[4];
ry(0.12800257323731223) q[5];
cx q[4],q[5];
ry(-2.7029382755738762) q[6];
ry(-1.725830839447448) q[7];
cx q[6],q[7];
ry(0.3564523261591235) q[6];
ry(2.807047676516361) q[7];
cx q[6],q[7];
ry(1.7865250214786013) q[0];
ry(2.6681936002483853) q[2];
cx q[0],q[2];
ry(-1.7069582650342752) q[0];
ry(3.1388747819583664) q[2];
cx q[0],q[2];
ry(-0.7748602852514079) q[2];
ry(1.4288671793131087) q[4];
cx q[2],q[4];
ry(-2.360902580997933) q[2];
ry(-0.9505409673028996) q[4];
cx q[2],q[4];
ry(-1.2431223413547752) q[4];
ry(-2.3810401789926923) q[6];
cx q[4],q[6];
ry(1.217005447446704) q[4];
ry(1.3644866089003198) q[6];
cx q[4],q[6];
ry(0.6789846471794264) q[1];
ry(3.0098376770493758) q[3];
cx q[1],q[3];
ry(-1.467090429839172) q[1];
ry(-2.502628149008507) q[3];
cx q[1],q[3];
ry(-0.6495168362106352) q[3];
ry(-2.6912031553178446) q[5];
cx q[3],q[5];
ry(1.5958675247150476) q[3];
ry(0.2135980671570401) q[5];
cx q[3],q[5];
ry(0.7900997836090076) q[5];
ry(-0.4621797372658362) q[7];
cx q[5],q[7];
ry(2.7599054030505563) q[5];
ry(-2.436997321693719) q[7];
cx q[5],q[7];
ry(0.23706393628601463) q[0];
ry(1.4312283814302662) q[3];
cx q[0],q[3];
ry(-2.6870442747311785) q[0];
ry(-2.242174301697246) q[3];
cx q[0],q[3];
ry(1.4823432993179742) q[1];
ry(0.8720252398773161) q[2];
cx q[1],q[2];
ry(-1.9023243172515205) q[1];
ry(-1.845940698077532) q[2];
cx q[1],q[2];
ry(-3.118400456648844) q[2];
ry(-2.366467872170268) q[5];
cx q[2],q[5];
ry(0.8695524862167673) q[2];
ry(3.0736965193961905) q[5];
cx q[2],q[5];
ry(0.33064166201442485) q[3];
ry(-2.115981781963142) q[4];
cx q[3],q[4];
ry(-1.4656919081214577) q[3];
ry(-2.896716147696368) q[4];
cx q[3],q[4];
ry(2.6132917465635335) q[4];
ry(1.208613754396982) q[7];
cx q[4],q[7];
ry(1.887871339747294) q[4];
ry(0.118642797701521) q[7];
cx q[4],q[7];
ry(0.08579110293975756) q[5];
ry(2.421468182727915) q[6];
cx q[5],q[6];
ry(1.4048207178837566) q[5];
ry(2.3959189882243543) q[6];
cx q[5],q[6];
ry(-0.7514637117815868) q[0];
ry(2.4490859587643716) q[1];
cx q[0],q[1];
ry(0.9211557253773419) q[0];
ry(0.27177128840817527) q[1];
cx q[0],q[1];
ry(-3.0149756091450945) q[2];
ry(-0.49815076816747617) q[3];
cx q[2],q[3];
ry(-1.4597645571835933) q[2];
ry(-0.9959924828303794) q[3];
cx q[2],q[3];
ry(1.7168975962157331) q[4];
ry(-2.7075052236563506) q[5];
cx q[4],q[5];
ry(-1.906092960779914) q[4];
ry(2.201066526932114) q[5];
cx q[4],q[5];
ry(0.8973460955285429) q[6];
ry(-1.0645815831924637) q[7];
cx q[6],q[7];
ry(-0.9204198928349095) q[6];
ry(-2.115695867404619) q[7];
cx q[6],q[7];
ry(2.677667403331982) q[0];
ry(1.8471976389249456) q[2];
cx q[0],q[2];
ry(-3.1100239545126773) q[0];
ry(0.5343773404917597) q[2];
cx q[0],q[2];
ry(-2.4196829841916228) q[2];
ry(-1.2160683243218005) q[4];
cx q[2],q[4];
ry(2.0978859943570316) q[2];
ry(-1.3188530749365013) q[4];
cx q[2],q[4];
ry(-0.6356368699176437) q[4];
ry(-0.5907909461471932) q[6];
cx q[4],q[6];
ry(-0.4407331533054996) q[4];
ry(0.5540800924171307) q[6];
cx q[4],q[6];
ry(-1.8661494373745948) q[1];
ry(-1.9570605191004988) q[3];
cx q[1],q[3];
ry(2.3721467958425366) q[1];
ry(-2.48996270840485) q[3];
cx q[1],q[3];
ry(-0.14018341518280555) q[3];
ry(-2.114296163125724) q[5];
cx q[3],q[5];
ry(-2.207732759750388) q[3];
ry(2.82891142973874) q[5];
cx q[3],q[5];
ry(0.2971954788839106) q[5];
ry(0.5636654153581817) q[7];
cx q[5],q[7];
ry(1.0673292490743105) q[5];
ry(-1.9987532696171573) q[7];
cx q[5],q[7];
ry(0.039588063545428404) q[0];
ry(2.7749061087100513) q[3];
cx q[0],q[3];
ry(1.5722321820828595) q[0];
ry(-0.04908877876009577) q[3];
cx q[0],q[3];
ry(0.6412389907520915) q[1];
ry(-1.3774053898903942) q[2];
cx q[1],q[2];
ry(2.949656595530507) q[1];
ry(-3.084228892622351) q[2];
cx q[1],q[2];
ry(-1.2748012675397997) q[2];
ry(2.956230539669532) q[5];
cx q[2],q[5];
ry(-0.4177123072604747) q[2];
ry(2.7629671582881503) q[5];
cx q[2],q[5];
ry(0.8792217877065052) q[3];
ry(1.2533865597943779) q[4];
cx q[3],q[4];
ry(-2.1706043706014713) q[3];
ry(1.9961451689978738) q[4];
cx q[3],q[4];
ry(-2.2847246342356056) q[4];
ry(3.0436298718674357) q[7];
cx q[4],q[7];
ry(-1.2897522706669007) q[4];
ry(0.572992765412855) q[7];
cx q[4],q[7];
ry(1.787913803136629) q[5];
ry(-2.843688993094998) q[6];
cx q[5],q[6];
ry(-2.0539949857697097) q[5];
ry(-0.9194982020859834) q[6];
cx q[5],q[6];
ry(0.04731687958079747) q[0];
ry(-2.3256956674582003) q[1];
cx q[0],q[1];
ry(-1.2847554619106802) q[0];
ry(2.850828148530985) q[1];
cx q[0],q[1];
ry(-0.7292376073213696) q[2];
ry(-2.620850131532544) q[3];
cx q[2],q[3];
ry(1.5864819731339415) q[2];
ry(-0.5009764430169819) q[3];
cx q[2],q[3];
ry(-1.8363451035709868) q[4];
ry(0.5923659400684107) q[5];
cx q[4],q[5];
ry(-2.1017065512723585) q[4];
ry(2.5797762029298483) q[5];
cx q[4],q[5];
ry(-1.3956451092703528) q[6];
ry(1.0969497587535582) q[7];
cx q[6],q[7];
ry(2.8501202634379936) q[6];
ry(3.1279627790387083) q[7];
cx q[6],q[7];
ry(2.6586257895037835) q[0];
ry(0.713958267635843) q[2];
cx q[0],q[2];
ry(-1.9818276408388904) q[0];
ry(3.049258643570813) q[2];
cx q[0],q[2];
ry(1.5350105279657498) q[2];
ry(1.4823503298784622) q[4];
cx q[2],q[4];
ry(-1.6540071754968722) q[2];
ry(1.9911848723747625) q[4];
cx q[2],q[4];
ry(-0.2563316945302727) q[4];
ry(0.6623751257158439) q[6];
cx q[4],q[6];
ry(-1.5418251840110262) q[4];
ry(-1.6673708039616932) q[6];
cx q[4],q[6];
ry(-2.8241021557245523) q[1];
ry(-0.9369342213221286) q[3];
cx q[1],q[3];
ry(-1.5897297820887317) q[1];
ry(2.1170636893276527) q[3];
cx q[1],q[3];
ry(2.021929521468433) q[3];
ry(2.032393032516161) q[5];
cx q[3],q[5];
ry(-0.102448273885833) q[3];
ry(2.636998459224625) q[5];
cx q[3],q[5];
ry(0.21877348302635458) q[5];
ry(2.042686540786771) q[7];
cx q[5],q[7];
ry(1.697449532384794) q[5];
ry(-1.3600169364231485) q[7];
cx q[5],q[7];
ry(0.002242038068309463) q[0];
ry(-0.1995379405285308) q[3];
cx q[0],q[3];
ry(-0.27094776782429725) q[0];
ry(0.3919278324170481) q[3];
cx q[0],q[3];
ry(-1.641487675585757) q[1];
ry(-2.2613370171276737) q[2];
cx q[1],q[2];
ry(-2.436387801528988) q[1];
ry(1.8484587743506236) q[2];
cx q[1],q[2];
ry(-2.097652495055512) q[2];
ry(0.3371040778123948) q[5];
cx q[2],q[5];
ry(-2.482396200087531) q[2];
ry(-0.7689036672869253) q[5];
cx q[2],q[5];
ry(-2.147316793046809) q[3];
ry(0.5263178883580776) q[4];
cx q[3],q[4];
ry(2.731640392404326) q[3];
ry(0.8769708261056984) q[4];
cx q[3],q[4];
ry(-2.3635823019458178) q[4];
ry(0.006915696912028757) q[7];
cx q[4],q[7];
ry(-1.1443939186218945) q[4];
ry(0.08343379896837509) q[7];
cx q[4],q[7];
ry(-0.17090758575402926) q[5];
ry(1.431950875628949) q[6];
cx q[5],q[6];
ry(0.9339391985674578) q[5];
ry(2.7574591704745575) q[6];
cx q[5],q[6];
ry(-1.787995209829986) q[0];
ry(-2.9598684503835235) q[1];
cx q[0],q[1];
ry(0.8664345632837789) q[0];
ry(1.027921780766536) q[1];
cx q[0],q[1];
ry(2.161097776622401) q[2];
ry(-2.162077593942334) q[3];
cx q[2],q[3];
ry(-1.9397526874012572) q[2];
ry(2.233863554588951) q[3];
cx q[2],q[3];
ry(-0.2729475397366513) q[4];
ry(-3.0780445296163275) q[5];
cx q[4],q[5];
ry(2.8629323548783003) q[4];
ry(0.6482195768473927) q[5];
cx q[4],q[5];
ry(2.248863584239105) q[6];
ry(-2.6332175629022445) q[7];
cx q[6],q[7];
ry(-2.463801764650609) q[6];
ry(2.0029361547274718) q[7];
cx q[6],q[7];
ry(1.7519256299896) q[0];
ry(0.6533801777442562) q[2];
cx q[0],q[2];
ry(-2.079384652305375) q[0];
ry(1.0770963571791317) q[2];
cx q[0],q[2];
ry(-1.411350811673585) q[2];
ry(0.6688284352561968) q[4];
cx q[2],q[4];
ry(-1.2877700989405882) q[2];
ry(-2.4689957088170025) q[4];
cx q[2],q[4];
ry(1.5595625614051063) q[4];
ry(2.6767051147848173) q[6];
cx q[4],q[6];
ry(-0.7553083813988969) q[4];
ry(2.9889403250768334) q[6];
cx q[4],q[6];
ry(1.9222449551584895) q[1];
ry(0.7844465088153658) q[3];
cx q[1],q[3];
ry(-0.45348557545789525) q[1];
ry(-0.7979793697416399) q[3];
cx q[1],q[3];
ry(0.30504103144722544) q[3];
ry(2.5856467348050547) q[5];
cx q[3],q[5];
ry(-2.0281110314696438) q[3];
ry(-0.9829732973457578) q[5];
cx q[3],q[5];
ry(0.16091045987578131) q[5];
ry(0.3261529817938913) q[7];
cx q[5],q[7];
ry(-0.32241044369740113) q[5];
ry(0.7507375002808137) q[7];
cx q[5],q[7];
ry(-1.1082228739950635) q[0];
ry(-0.2714376399766492) q[3];
cx q[0],q[3];
ry(3.014391587385956) q[0];
ry(-2.969436579857365) q[3];
cx q[0],q[3];
ry(-2.7131124562715168) q[1];
ry(-0.20691082283005727) q[2];
cx q[1],q[2];
ry(-3.0514974017238004) q[1];
ry(-1.480168239130441) q[2];
cx q[1],q[2];
ry(-0.4989867939167177) q[2];
ry(2.1710766582549126) q[5];
cx q[2],q[5];
ry(0.8816981347649248) q[2];
ry(-3.033363020132741) q[5];
cx q[2],q[5];
ry(2.6796565926093385) q[3];
ry(1.3104733646396822) q[4];
cx q[3],q[4];
ry(-2.4033762658911297) q[3];
ry(0.45370774310251516) q[4];
cx q[3],q[4];
ry(2.6534497947615234) q[4];
ry(-0.7511723298314514) q[7];
cx q[4],q[7];
ry(0.0033377207223144268) q[4];
ry(2.6287475252161094) q[7];
cx q[4],q[7];
ry(2.5158098027303093) q[5];
ry(-0.7499327919079666) q[6];
cx q[5],q[6];
ry(-0.4355394131466719) q[5];
ry(-0.14347061251848284) q[6];
cx q[5],q[6];
ry(0.19830669903032647) q[0];
ry(-3.1267444548004084) q[1];
cx q[0],q[1];
ry(-0.8608435035348252) q[0];
ry(3.099544286813711) q[1];
cx q[0],q[1];
ry(1.299765406934841) q[2];
ry(-2.609068216794994) q[3];
cx q[2],q[3];
ry(-1.48307960948814) q[2];
ry(-1.470345922283305) q[3];
cx q[2],q[3];
ry(1.9546740403620442) q[4];
ry(1.8130628129745414) q[5];
cx q[4],q[5];
ry(-1.7064790844284714) q[4];
ry(-1.3839544111589968) q[5];
cx q[4],q[5];
ry(-2.383764063878227) q[6];
ry(-0.08366194988266457) q[7];
cx q[6],q[7];
ry(3.1294675474936042) q[6];
ry(-0.1425151081450906) q[7];
cx q[6],q[7];
ry(-0.037377521001031186) q[0];
ry(1.0972522474142723) q[2];
cx q[0],q[2];
ry(-1.917102548567773) q[0];
ry(-2.5232876754249105) q[2];
cx q[0],q[2];
ry(1.1327689525525866) q[2];
ry(-1.4848487964146615) q[4];
cx q[2],q[4];
ry(1.8168925034383765) q[2];
ry(-2.609445655020376) q[4];
cx q[2],q[4];
ry(1.1447615496626693) q[4];
ry(0.3269766695125818) q[6];
cx q[4],q[6];
ry(1.6695787105416022) q[4];
ry(-1.0217323190355205) q[6];
cx q[4],q[6];
ry(-2.399643590020563) q[1];
ry(-0.9861801545549325) q[3];
cx q[1],q[3];
ry(1.6342386486207268) q[1];
ry(-1.0834284277472293) q[3];
cx q[1],q[3];
ry(0.733538050173928) q[3];
ry(-2.2368095975564133) q[5];
cx q[3],q[5];
ry(1.7109915275410872) q[3];
ry(1.8441550189169895) q[5];
cx q[3],q[5];
ry(-2.7112784782057697) q[5];
ry(-3.0331755052002882) q[7];
cx q[5],q[7];
ry(-1.6301700098909129) q[5];
ry(-1.5964848474277136) q[7];
cx q[5],q[7];
ry(-2.480849445528524) q[0];
ry(1.16969279281407) q[3];
cx q[0],q[3];
ry(1.449230303233767) q[0];
ry(2.446137251794066) q[3];
cx q[0],q[3];
ry(1.8830120562494423) q[1];
ry(-2.9253577099910726) q[2];
cx q[1],q[2];
ry(-0.4527117875586922) q[1];
ry(-0.39004641667420914) q[2];
cx q[1],q[2];
ry(-1.6436739706490542) q[2];
ry(-0.5633132035097178) q[5];
cx q[2],q[5];
ry(1.7746305324655003) q[2];
ry(-2.8605500671841932) q[5];
cx q[2],q[5];
ry(0.31403579032665396) q[3];
ry(-2.1123561150090895) q[4];
cx q[3],q[4];
ry(-0.6976818459252838) q[3];
ry(-0.8584251930598964) q[4];
cx q[3],q[4];
ry(1.6161189490013956) q[4];
ry(0.031964562578578735) q[7];
cx q[4],q[7];
ry(0.7443135312444044) q[4];
ry(0.18202648331603735) q[7];
cx q[4],q[7];
ry(-0.9325973059471018) q[5];
ry(-1.7278864341369626) q[6];
cx q[5],q[6];
ry(-2.3813765787102605) q[5];
ry(-2.759991449070802) q[6];
cx q[5],q[6];
ry(1.8581048255306418) q[0];
ry(-2.1885368788700186) q[1];
cx q[0],q[1];
ry(-1.641100128498021) q[0];
ry(0.584629224567925) q[1];
cx q[0],q[1];
ry(-0.9554807250009189) q[2];
ry(-1.905634878029982) q[3];
cx q[2],q[3];
ry(-2.4138896163524723) q[2];
ry(1.6229521617063973) q[3];
cx q[2],q[3];
ry(1.930846508813186) q[4];
ry(-0.3182117694490552) q[5];
cx q[4],q[5];
ry(0.5697792671839812) q[4];
ry(2.640547584741685) q[5];
cx q[4],q[5];
ry(2.860748234640593) q[6];
ry(-0.05771327652163878) q[7];
cx q[6],q[7];
ry(0.0774290988785824) q[6];
ry(-2.764573702696902) q[7];
cx q[6],q[7];
ry(0.5303845776322105) q[0];
ry(2.4554254419227104) q[2];
cx q[0],q[2];
ry(1.8481076561556604) q[0];
ry(2.993580730202425) q[2];
cx q[0],q[2];
ry(2.28447812058103) q[2];
ry(2.6698199485529606) q[4];
cx q[2],q[4];
ry(3.086644880798545) q[2];
ry(0.8912046816770092) q[4];
cx q[2],q[4];
ry(0.7355556136960132) q[4];
ry(2.7530265652634616) q[6];
cx q[4],q[6];
ry(-1.9932742276344637) q[4];
ry(-0.8192034423607287) q[6];
cx q[4],q[6];
ry(0.0004908121936261334) q[1];
ry(-2.4624036037502126) q[3];
cx q[1],q[3];
ry(0.5716065632785265) q[1];
ry(-2.482587749954375) q[3];
cx q[1],q[3];
ry(2.4976615070348944) q[3];
ry(3.1295164598946403) q[5];
cx q[3],q[5];
ry(-1.552265469649732) q[3];
ry(1.2125289724352095) q[5];
cx q[3],q[5];
ry(-2.706352751533492) q[5];
ry(-1.5254719935165637) q[7];
cx q[5],q[7];
ry(2.337226285673401) q[5];
ry(1.5894337375240895) q[7];
cx q[5],q[7];
ry(2.8392895136468423) q[0];
ry(2.69992505017657) q[3];
cx q[0],q[3];
ry(1.1696986252968158) q[0];
ry(2.561870342226201) q[3];
cx q[0],q[3];
ry(1.8339319873009117) q[1];
ry(1.945455938043912) q[2];
cx q[1],q[2];
ry(-0.4295416172322853) q[1];
ry(-2.422644064971231) q[2];
cx q[1],q[2];
ry(-1.4437196156873748) q[2];
ry(-3.0011097337003285) q[5];
cx q[2],q[5];
ry(-1.740148726525831) q[2];
ry(0.4241949417041107) q[5];
cx q[2],q[5];
ry(-1.2095207021919592) q[3];
ry(1.9998598738828353) q[4];
cx q[3],q[4];
ry(-0.8062365729226093) q[3];
ry(0.9613862841308387) q[4];
cx q[3],q[4];
ry(2.325828599891304) q[4];
ry(0.12207641988632734) q[7];
cx q[4],q[7];
ry(-2.2934359233792105) q[4];
ry(-1.7720236418673192) q[7];
cx q[4],q[7];
ry(-2.680630151249159) q[5];
ry(-0.3661830520988616) q[6];
cx q[5],q[6];
ry(2.0124669738007483) q[5];
ry(-1.7824369019452029) q[6];
cx q[5],q[6];
ry(-1.8961248629650074) q[0];
ry(-0.4015166120894591) q[1];
ry(2.8757043923446304) q[2];
ry(-1.1299806075248533) q[3];
ry(-1.8277948810759768) q[4];
ry(-0.5508843622960313) q[5];
ry(3.0600914350472674) q[6];
ry(-2.49439420443622) q[7];