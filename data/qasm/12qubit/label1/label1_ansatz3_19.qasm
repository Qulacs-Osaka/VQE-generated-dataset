OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(2.639425224567565) q[0];
rz(1.6957101247953852) q[0];
ry(2.2477885061209917) q[1];
rz(-2.5449145877444606) q[1];
ry(1.2418427133216878) q[2];
rz(1.1018712488587903) q[2];
ry(0.07377135970102675) q[3];
rz(0.9238701157365499) q[3];
ry(-2.8068664998739608) q[4];
rz(0.7831980262866125) q[4];
ry(0.36267603028337353) q[5];
rz(-0.29881193784721655) q[5];
ry(2.7110879320967456) q[6];
rz(2.42661892971302) q[6];
ry(-2.4705736705535726) q[7];
rz(0.6388145736860347) q[7];
ry(0.1270366719128801) q[8];
rz(-2.735381268460682) q[8];
ry(3.1073426634212953) q[9];
rz(2.3473782912152523) q[9];
ry(-1.9642365194798774) q[10];
rz(-0.8883805378423961) q[10];
ry(-1.8390139033945414) q[11];
rz(2.6338720428968303) q[11];
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
ry(1.5678189204634918) q[0];
rz(0.320687904651888) q[0];
ry(-2.2274603320459168) q[1];
rz(-0.4197753177002382) q[1];
ry(0.9487684670221364) q[2];
rz(-2.2799914548021247) q[2];
ry(-0.03398466653980926) q[3];
rz(-0.8750198276139987) q[3];
ry(0.0029250293054500537) q[4];
rz(0.6915107533828229) q[4];
ry(-0.07113573668313801) q[5];
rz(0.016528604343637454) q[5];
ry(-2.971751242470646) q[6];
rz(-2.1164554834749456) q[6];
ry(-2.0569834652491776) q[7];
rz(-0.8274179309638061) q[7];
ry(-0.06869588965933578) q[8];
rz(2.1653479373111075) q[8];
ry(-3.056388576359086) q[9];
rz(1.6373618139355959) q[9];
ry(0.6018289605572296) q[10];
rz(1.8189712529724078) q[10];
ry(2.881190874209851) q[11];
rz(1.2136994733227637) q[11];
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
ry(-3.0886952405327435) q[0];
rz(-2.194790971460689) q[0];
ry(2.1759733979625366) q[1];
rz(-2.1531103114651406) q[1];
ry(0.9113729382555755) q[2];
rz(-1.5035739752405988) q[2];
ry(-0.8288965850289824) q[3];
rz(2.8059010169157346) q[3];
ry(3.1119620988297334) q[4];
rz(2.2363907548802997) q[4];
ry(-1.602486510636877) q[5];
rz(2.2424706837100534) q[5];
ry(-1.1295250798265368) q[6];
rz(-0.47416937040909957) q[6];
ry(0.192371951031566) q[7];
rz(0.8761166538108434) q[7];
ry(2.993935633996048) q[8];
rz(-1.8864441140425834) q[8];
ry(3.052965749286592) q[9];
rz(-1.0641312910659826) q[9];
ry(0.009739558631683017) q[10];
rz(-2.186508958270091) q[10];
ry(0.6444382520535585) q[11];
rz(-0.9572624330314085) q[11];
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
ry(-2.08230788257738) q[0];
rz(-0.6693155457692286) q[0];
ry(-1.0898611066056576) q[1];
rz(1.899269041171726) q[1];
ry(0.9097598855649309) q[2];
rz(-0.588698123854746) q[2];
ry(-0.0013973834168735711) q[3];
rz(3.0836671662272477) q[3];
ry(-0.0003157916683175658) q[4];
rz(2.6369646696580635) q[4];
ry(2.8560590159149903) q[5];
rz(-2.1902882235472236) q[5];
ry(0.1664383532945184) q[6];
rz(3.011651648987529) q[6];
ry(2.822877190063423) q[7];
rz(2.7461289384231016) q[7];
ry(2.0618502743765523) q[8];
rz(2.0853515158019933) q[8];
ry(-0.08967893246538415) q[9];
rz(2.7335876885777806) q[9];
ry(1.1716184517812724) q[10];
rz(-2.1655940105692792) q[10];
ry(2.6349555073086903) q[11];
rz(2.7403163397201906) q[11];
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
ry(2.2815879828282366) q[0];
rz(-1.638213563627052) q[0];
ry(-2.9072889634716765) q[1];
rz(1.930477650481643) q[1];
ry(-1.7161234347479306) q[2];
rz(-1.451099629082379) q[2];
ry(-3.119418537802107) q[3];
rz(-1.4578365257254093) q[3];
ry(-3.1330552132737184) q[4];
rz(0.8200901896662991) q[4];
ry(-1.5799519494579801) q[5];
rz(0.9112915181718995) q[5];
ry(3.0609327927236474) q[6];
rz(-2.906310571720528) q[6];
ry(1.5383407049491513) q[7];
rz(-1.262622584339093) q[7];
ry(-3.102635975129267) q[8];
rz(3.0161153779482057) q[8];
ry(1.6115202084134617) q[9];
rz(-2.4455700404053275) q[9];
ry(-3.0670642498092695) q[10];
rz(-1.2918506103021485) q[10];
ry(-2.1949483143046598) q[11];
rz(-2.8751329605119853) q[11];
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
ry(2.007188185018782) q[0];
rz(2.730415598006822) q[0];
ry(0.6790685959452647) q[1];
rz(-1.5980143077869302) q[1];
ry(1.7545690260252798) q[2];
rz(2.6189777017618354) q[2];
ry(3.1137429919336004) q[3];
rz(-2.8207387381414963) q[3];
ry(3.1380124697219927) q[4];
rz(1.3462928718726186) q[4];
ry(2.20225745815507) q[5];
rz(0.9340187083489412) q[5];
ry(-3.1272985944560845) q[6];
rz(0.9005722469074663) q[6];
ry(3.1100571973746534) q[7];
rz(0.5996422957813896) q[7];
ry(2.6652812288669048) q[8];
rz(0.44211682913966227) q[8];
ry(2.981111283597089) q[9];
rz(1.6358575916401288) q[9];
ry(2.648310888198699) q[10];
rz(-1.5819035773479728) q[10];
ry(-1.6559579184824658) q[11];
rz(0.6880907707866992) q[11];
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
ry(1.8145851950355116) q[0];
rz(-0.5964253713818332) q[0];
ry(0.0288841189899216) q[1];
rz(-3.031974682281761) q[1];
ry(1.662269768246306) q[2];
rz(1.5738567423217802) q[2];
ry(3.1217789980147557) q[3];
rz(-3.0493436031724523) q[3];
ry(-1.5876128492147057) q[4];
rz(1.5816025373487776) q[4];
ry(2.328306438509127) q[5];
rz(0.23701861648392034) q[5];
ry(0.0826791510640678) q[6];
rz(-0.8567326061742677) q[6];
ry(0.009315735340639897) q[7];
rz(-0.4958547226934171) q[7];
ry(-2.4497378143380883) q[8];
rz(-3.040823301869148) q[8];
ry(1.4814723674317805) q[9];
rz(-2.2761815996658443) q[9];
ry(-3.0397883808504087) q[10];
rz(0.48188642420517963) q[10];
ry(1.5314495574462057) q[11];
rz(0.1179111905664545) q[11];
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
ry(-1.4143411680210882) q[0];
rz(1.0345864556687678) q[0];
ry(0.6125730992474026) q[1];
rz(1.9420978220496348) q[1];
ry(-0.015921071932672823) q[2];
rz(2.08577041355818) q[2];
ry(0.0005901446278357979) q[3];
rz(2.4397854932047682) q[3];
ry(-3.1400555837070105) q[4];
rz(-1.5814428373632878) q[4];
ry(-1.5814241971435976) q[5];
rz(1.5736721604676847) q[5];
ry(-3.141156700332472) q[6];
rz(0.8211177389143155) q[6];
ry(0.0473704119745042) q[7];
rz(1.7446323547749296) q[7];
ry(-1.6754477347812387) q[8];
rz(-2.706709957766876) q[8];
ry(-3.098476578938797) q[9];
rz(-2.4800869144968694) q[9];
ry(-0.2247541679730052) q[10];
rz(1.5837201318619005) q[10];
ry(-0.2629726912721768) q[11];
rz(-1.6177720564185307) q[11];
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
ry(-0.670908871109009) q[0];
rz(-2.893737955807359) q[0];
ry(-0.053283884468327436) q[1];
rz(-2.13207476903314) q[1];
ry(-0.9872666579532448) q[2];
rz(-1.8651357520766947) q[2];
ry(-0.394426234496998) q[3];
rz(2.311219451296753) q[3];
ry(-2.1906408239769926) q[4];
rz(1.0098923082078015) q[4];
ry(-1.5745746684102215) q[5];
rz(0.5543245992499708) q[5];
ry(1.419744950779644) q[6];
rz(3.039589776328023) q[6];
ry(1.5724371888107962) q[7];
rz(-1.7072541645061317) q[7];
ry(2.225830805137381) q[8];
rz(-1.8074790774486342) q[8];
ry(-1.1634260329788266) q[9];
rz(1.7208376317676792) q[9];
ry(1.4086067351482177) q[10];
rz(-0.9968205690259246) q[10];
ry(-2.293151082065998) q[11];
rz(-2.6776588463865414) q[11];
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
ry(-2.3915046250094587) q[0];
rz(-1.2076864204006554) q[0];
ry(-1.2023705751159868) q[1];
rz(-1.3153145091361855) q[1];
ry(3.0393215339004804) q[2];
rz(0.42674058044813806) q[2];
ry(3.1361389029752575) q[3];
rz(-1.315888515710614) q[3];
ry(-0.00024875843876886165) q[4];
rz(0.5867127535058769) q[4];
ry(1.454427492606543) q[5];
rz(-1.137023767330966) q[5];
ry(-0.06497377407385085) q[6];
rz(-3.039857509562311) q[6];
ry(3.069831790902176) q[7];
rz(-0.11229832422183872) q[7];
ry(-2.9861726330151823) q[8];
rz(1.8573648801043916) q[8];
ry(-1.566316797521478) q[9];
rz(-1.5725169308279918) q[9];
ry(-0.12897355197449745) q[10];
rz(2.873607824564014) q[10];
ry(0.03963755124282962) q[11];
rz(1.0013781351778164) q[11];
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
ry(-1.7328419273251976) q[0];
rz(1.4466679522709338) q[0];
ry(-2.2113091598744212) q[1];
rz(-0.06238974838921637) q[1];
ry(-2.158962803630782) q[2];
rz(0.0626887974979241) q[2];
ry(3.136672722638092) q[3];
rz(2.0499048452773874) q[3];
ry(-3.026947351983801) q[4];
rz(2.067014370558483) q[4];
ry(-0.004863294847058247) q[5];
rz(2.22648000044744) q[5];
ry(-0.642131524122334) q[6];
rz(-2.82163961658399) q[6];
ry(3.1345949972342737) q[7];
rz(-1.5487661382171147) q[7];
ry(-1.5737007882055938) q[8];
rz(-1.1397959744773445) q[8];
ry(-1.612565272728487) q[9];
rz(3.0812750942492815) q[9];
ry(0.18240509346580058) q[10];
rz(-1.4111179449664073) q[10];
ry(-0.0020337579816895575) q[11];
rz(0.5180726222553363) q[11];
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
ry(1.0737225204999166) q[0];
rz(2.556891126065931) q[0];
ry(-1.487135197112333) q[1];
rz(0.9513433043863024) q[1];
ry(3.0849349195832136) q[2];
rz(-3.11205431749495) q[2];
ry(0.004643469132079581) q[3];
rz(2.4258586893032454) q[3];
ry(3.141281309171444) q[4];
rz(2.020664109873395) q[4];
ry(2.9451980423345843) q[5];
rz(-1.0360009095960292) q[5];
ry(-0.00039244051042423963) q[6];
rz(2.313403073117824) q[6];
ry(2.8440532946807227) q[7];
rz(0.9896011613628372) q[7];
ry(-0.0045412536733993) q[8];
rz(2.7115971761388806) q[8];
ry(-1.5678139099399857) q[9];
rz(1.3419738996012727) q[9];
ry(-3.1393026985169015) q[10];
rz(-1.4458045912452804) q[10];
ry(-3.1404972904767514) q[11];
rz(0.23303876731377685) q[11];
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
ry(-0.47927809438136837) q[0];
rz(-0.25745042397560614) q[0];
ry(-1.5688558585390031) q[1];
rz(1.1452990577739104) q[1];
ry(0.8900975609400925) q[2];
rz(0.9202694685616155) q[2];
ry(-0.00016445677784737714) q[3];
rz(0.4805708707532999) q[3];
ry(-0.10619949399650341) q[4];
rz(-0.04170624017751699) q[4];
ry(-0.004789208719915727) q[5];
rz(2.114026836880103) q[5];
ry(2.2934767199346755) q[6];
rz(-1.5363796771477969) q[6];
ry(-0.0013064379428989123) q[7];
rz(2.1452309014816757) q[7];
ry(-1.5694744810495713) q[8];
rz(-0.4162885522099519) q[8];
ry(-0.19551484472013847) q[9];
rz(1.8076903487798641) q[9];
ry(0.5370079427156647) q[10];
rz(-0.0859809085374174) q[10];
ry(-0.007485222701434147) q[11];
rz(0.9182735260717703) q[11];
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
ry(-1.721664089776818) q[0];
rz(-2.733749152973653) q[0];
ry(-2.326603065645578) q[1];
rz(-0.9492626056512874) q[1];
ry(-2.366988213073953) q[2];
rz(-3.0771168850800943) q[2];
ry(-3.140991615466992) q[3];
rz(2.5869935556307264) q[3];
ry(-0.0005255028205493063) q[4];
rz(-0.7974207552972445) q[4];
ry(1.570016760557742) q[5];
rz(-2.954435634240057) q[5];
ry(0.1104220740607378) q[6];
rz(-0.3808978350594714) q[6];
ry(-2.910385325223935) q[7];
rz(1.5746572263292222) q[7];
ry(-3.0479223200216845) q[8];
rz(-0.16427086269511815) q[8];
ry(0.8871742546385826) q[9];
rz(-0.003341607170016303) q[9];
ry(1.0389234141865165) q[10];
rz(-3.025891246842265) q[10];
ry(-3.1395952518050274) q[11];
rz(-2.8877641272519634) q[11];
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
ry(-0.08703300780353454) q[0];
rz(-1.2172620243418226) q[0];
ry(2.7553133149484945) q[1];
rz(-2.7715496763332053) q[1];
ry(-1.2734931531054459) q[2];
rz(-3.0633944787956047) q[2];
ry(-3.0933675775061866) q[3];
rz(-2.5731846523200765) q[3];
ry(0.04031590471071489) q[4];
rz(1.2773622754545044) q[4];
ry(1.7711490816716893) q[5];
rz(0.946339819444708) q[5];
ry(-0.34828368650300956) q[6];
rz(-3.0371054334443333) q[6];
ry(2.9301433698510175) q[7];
rz(-1.4310488770724223) q[7];
ry(-0.0023160540442921373) q[8];
rz(-2.7476494928943676) q[8];
ry(-1.4850138716726518) q[9];
rz(-1.9817144806207312) q[9];
ry(2.290835254435316) q[10];
rz(-2.2422270366324266) q[10];
ry(2.54438801458602) q[11];
rz(2.1180637983281616) q[11];
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
ry(-1.7299732796945853) q[0];
rz(-3.0661448007017054) q[0];
ry(1.251513233691579) q[1];
rz(-1.4319586836577871) q[1];
ry(0.8179712297561136) q[2];
rz(-1.8845295114723644) q[2];
ry(3.1240417719182574) q[3];
rz(-1.8022192991167454) q[3];
ry(-0.005302611036622551) q[4];
rz(1.1901109525638658) q[4];
ry(-3.096713073343122) q[5];
rz(2.608269241881394) q[5];
ry(-0.13374321260734928) q[6];
rz(1.4187362455719403) q[6];
ry(3.1106386076504786) q[7];
rz(1.694214259838974) q[7];
ry(1.5693769325342357) q[8];
rz(-1.6547301375855412) q[8];
ry(0.06557856930060237) q[9];
rz(-1.1846898609135232) q[9];
ry(1.9523910744477082) q[10];
rz(0.7451565989965685) q[10];
ry(0.0022797299799242055) q[11];
rz(-0.20073222544398475) q[11];
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
ry(-1.3730398208138999) q[0];
rz(1.1757510000920952) q[0];
ry(1.0688577948655564) q[1];
rz(2.0612890084442457) q[1];
ry(2.157198590241649) q[2];
rz(0.33445510236163234) q[2];
ry(-3.113477691320328) q[3];
rz(1.1463000710344462) q[3];
ry(2.1739644838631094) q[4];
rz(-2.987841643045778) q[4];
ry(0.7142177210911067) q[5];
rz(-0.05434408716535621) q[5];
ry(-0.0013526775017052017) q[6];
rz(-2.991406315723951) q[6];
ry(-1.0651446311893373) q[7];
rz(0.23148927792255525) q[7];
ry(3.1077226435626093) q[8];
rz(3.055278790355886) q[8];
ry(-1.5727659901824016) q[9];
rz(-1.5357202696300103) q[9];
ry(-0.0018015414297360977) q[10];
rz(1.9623804530653806) q[10];
ry(0.5210392895322228) q[11];
rz(-2.5063629317662226) q[11];
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
ry(1.642127614041641) q[0];
rz(-1.8937249114657968) q[0];
ry(0.0868198551215667) q[1];
rz(-1.482180030691911) q[1];
ry(-3.0109443901382282) q[2];
rz(-0.7373461130313403) q[2];
ry(-0.002763480490245987) q[3];
rz(1.5558880707731833) q[3];
ry(-3.1410727902700106) q[4];
rz(-2.9664977306928937) q[4];
ry(0.018528698141230038) q[5];
rz(0.514345750326041) q[5];
ry(0.044577747400653484) q[6];
rz(-2.7579891253002162) q[6];
ry(-0.00015230077810012913) q[7];
rz(-0.2395637224679126) q[7];
ry(1.622283116697498) q[8];
rz(3.0518744626663437) q[8];
ry(2.3535500377532985) q[9];
rz(-2.6813197568407334) q[9];
ry(-1.2441631983038792) q[10];
rz(1.7323990156433071) q[10];
ry(-2.7976048439489394) q[11];
rz(0.29616352431915505) q[11];
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
ry(1.8409788802805425) q[0];
rz(-1.1361076344735155) q[0];
ry(-0.541215446579883) q[1];
rz(-2.3249110339630703) q[1];
ry(-2.9735115749096686) q[2];
rz(2.109868125604557) q[2];
ry(-0.9685213480989183) q[3];
rz(0.009419654930198362) q[3];
ry(1.1547405392043562) q[4];
rz(-1.4319218032399836) q[4];
ry(-3.0247918002814966) q[5];
rz(1.3074535481951957) q[5];
ry(-1.332353838050604) q[6];
rz(3.0227651787957495) q[6];
ry(1.4108302823336507) q[7];
rz(1.8598068930951075) q[7];
ry(-0.4325754108524418) q[8];
rz(0.197299258954567) q[8];
ry(-1.9649190940138244) q[9];
rz(-2.4348473589993476) q[9];
ry(-1.7997651024740202) q[10];
rz(3.130331449110838) q[10];
ry(0.1344459278401544) q[11];
rz(2.0129688451075984) q[11];
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
ry(-1.7536315596443153) q[0];
rz(2.4403412296733182) q[0];
ry(2.7505300558459456) q[1];
rz(-1.1104762391282605) q[1];
ry(-0.16259001383714108) q[2];
rz(0.03465174442088781) q[2];
ry(0.004018947118724191) q[3];
rz(-0.09223671202540906) q[3];
ry(-3.1282485183892135) q[4];
rz(1.638854848171989) q[4];
ry(-3.1344898236812857) q[5];
rz(-0.8549410172113787) q[5];
ry(0.028854552684830028) q[6];
rz(0.06088602037250634) q[6];
ry(-3.140904757770387) q[7];
rz(1.9117331409432887) q[7];
ry(0.0027679177749488915) q[8];
rz(-0.181642578057164) q[8];
ry(-2.8248041572298965) q[9];
rz(0.4551174233891064) q[9];
ry(-1.1046952931425098) q[10];
rz(-3.125074131702134) q[10];
ry(1.1999740788241198) q[11];
rz(0.8171269884626502) q[11];
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
ry(1.6729652674925897) q[0];
rz(2.019932234304985) q[0];
ry(-3.074123248146419) q[1];
rz(-1.6995813587612671) q[1];
ry(-1.2216640999285788) q[2];
rz(-0.11399733392609523) q[2];
ry(-1.5562163242892229) q[3];
rz(0.6063451727094099) q[3];
ry(2.195823546348835) q[4];
rz(-3.0825057800971982) q[4];
ry(-2.9488786315061684) q[5];
rz(-1.3856899743255715) q[5];
ry(-1.303774825571393) q[6];
rz(-3.025041706323763) q[6];
ry(-2.974173985442615) q[7];
rz(2.6383010090265286) q[7];
ry(0.5729150931467958) q[8];
rz(1.1348396426246241) q[8];
ry(0.005689341029510687) q[9];
rz(2.824841650504786) q[9];
ry(-2.5667847358496796) q[10];
rz(2.4071275219309234) q[10];
ry(1.2187407569540618) q[11];
rz(1.3593802560212538) q[11];
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
ry(1.3789463803242548) q[0];
rz(-2.3903298623858706) q[0];
ry(2.4673549342457304) q[1];
rz(-2.9505844727207866) q[1];
ry(-0.08205183064847876) q[2];
rz(-3.1411507891725448) q[2];
ry(3.1348962648578094) q[3];
rz(0.3846264965851178) q[3];
ry(0.000631734196366196) q[4];
rz(1.4337694222085537) q[4];
ry(3.1262041806542955) q[5];
rz(-1.4498469722084462) q[5];
ry(-3.087637672868039) q[6];
rz(2.2119989967850975) q[6];
ry(0.0007809856923435276) q[7];
rz(2.127347147140961) q[7];
ry(-3.1341529469683143) q[8];
rz(2.7234678525979055) q[8];
ry(3.1324875984679075) q[9];
rz(2.9556242389640848) q[9];
ry(0.0005155027619453967) q[10];
rz(-0.8246474116903162) q[10];
ry(-1.7890100280624308) q[11];
rz(-2.5576515734629135) q[11];
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
ry(0.22390275699508333) q[0];
rz(1.1633476484391418) q[0];
ry(1.5900822392644232) q[1];
rz(0.00446565160244372) q[1];
ry(-0.2602379729108959) q[2];
rz(0.21219227130846094) q[2];
ry(-0.5203439070376081) q[3];
rz(0.8188599106232919) q[3];
ry(1.5306972482022012) q[4];
rz(1.2217883462035517) q[4];
ry(1.5935361009426678) q[5];
rz(0.363439844800519) q[5];
ry(0.05960217602202178) q[6];
rz(-1.8554888494525252) q[6];
ry(-1.552090113600321) q[7];
rz(-1.8947824925096823) q[7];
ry(1.5794215586658569) q[8];
rz(-2.2457541640332424) q[8];
ry(-3.141382647574979) q[9];
rz(-2.9639178881592048) q[9];
ry(-1.5638522409358064) q[10];
rz(-0.10223517411663424) q[10];
ry(0.9013837363780745) q[11];
rz(1.3552305394596154) q[11];