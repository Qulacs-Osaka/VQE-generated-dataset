OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.013467789103805316) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.02893584143578604) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.04636890471611542) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.15134658194996287) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.05127317703324584) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.0038800566255042753) q[3];
cx q[2],q[3];
rx(-0.13619754799213737) q[0];
rz(-0.09976877192949289) q[0];
rx(-0.04582512560016616) q[1];
rz(-0.0532977068522571) q[1];
rx(-0.06950237063824063) q[2];
rz(-0.08996509142952046) q[2];
rx(-0.07326670002526617) q[3];
rz(-0.062172937655101596) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.06216411513313263) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.04054065434025621) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.07466957351613991) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.11608265898247623) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.02579913081208813) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.03380858696530993) q[3];
cx q[2],q[3];
rx(-0.22877615454052808) q[0];
rz(-0.08511838611138134) q[0];
rx(-0.07322659521368263) q[1];
rz(-0.10813340556456201) q[1];
rx(-0.02873225905283852) q[2];
rz(-0.07124042464389231) q[2];
rx(-0.04126744701443379) q[3];
rz(-0.035800603469325655) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0927857284001425) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.03131718664674401) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.05970731008257273) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.17871774968211596) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.011748406576416287) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.04237916546579579) q[3];
cx q[2],q[3];
rx(-0.19149955496407456) q[0];
rz(-0.046473802549348894) q[0];
rx(-0.005070359754886394) q[1];
rz(-0.1407553040710419) q[1];
rx(-0.059250204106196636) q[2];
rz(-0.10171825946214161) q[2];
rx(-0.1115611679115848) q[3];
rz(-0.061630079603607946) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.08496934736398677) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.00282201714217919) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.04579376820650372) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.20478290546637082) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.007845623813497404) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.043016019478398324) q[3];
cx q[2],q[3];
rx(-0.20919052481514325) q[0];
rz(-0.006420896254319159) q[0];
rx(-0.009412365081132973) q[1];
rz(-0.10842325928002797) q[1];
rx(-0.0663948179196028) q[2];
rz(-0.0874051252027674) q[2];
rx(-0.17985689764950497) q[3];
rz(-0.03980568966970528) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.15093017177525253) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.04716438770470728) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.04638966457215795) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.2257100337519322) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.005248143367033322) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.046044760581966805) q[3];
cx q[2],q[3];
rx(-0.23069807496633243) q[0];
rz(-0.05755176340963931) q[0];
rx(0.006165095470657623) q[1];
rz(-0.06984795802151) q[1];
rx(0.033746492578272176) q[2];
rz(-0.09613290204247238) q[2];
rx(-0.17793171420129908) q[3];
rz(0.028605236918348428) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.13864158582824618) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.05550618899128135) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.02792629397315787) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.19188818561490956) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.05157979570796067) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.024027993645002833) q[3];
cx q[2],q[3];
rx(-0.17805930802159875) q[0];
rz(-0.07639177924511435) q[0];
rx(-0.012947103038493727) q[1];
rz(-0.1134549116569872) q[1];
rx(-0.053978343376716496) q[2];
rz(-0.13858060793697247) q[2];
rx(-0.17591700044122202) q[3];
rz(-0.04816600070374542) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.16775932566381135) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.04527383483887938) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.06924907186764376) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.12763281763917464) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.0743108687203329) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.005472856308381175) q[3];
cx q[2],q[3];
rx(-0.2103648290116299) q[0];
rz(-0.011410391943161643) q[0];
rx(0.030128433910752674) q[1];
rz(-0.12735014744799986) q[1];
rx(-0.04912287170875297) q[2];
rz(-0.09049124780209075) q[2];
rx(-0.16796390198297736) q[3];
rz(-0.018186641366329095) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.10202192790032172) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.015907245452055564) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.053492182987567624) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.13406176070261033) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.03284253152892481) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.028161718638996545) q[3];
cx q[2],q[3];
rx(-0.19864025171153815) q[0];
rz(-0.020800610056558116) q[0];
rx(0.062496087100876216) q[1];
rz(-0.0602360634529963) q[1];
rx(-0.034242110051820575) q[2];
rz(-0.12723050841285385) q[2];
rx(-0.1944335168948914) q[3];
rz(-0.013630691042699468) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1342629662705826) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.035773006718845056) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.0846971685987319) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1341987296510882) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.00300928639515042) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.04854266088794078) q[3];
cx q[2],q[3];
rx(-0.24705323489899836) q[0];
rz(0.013600211286335243) q[0];
rx(-0.011652406861899984) q[1];
rz(-0.13412500844910388) q[1];
rx(-0.013483461646637608) q[2];
rz(-0.1355653535480352) q[2];
rx(-0.23585602629213967) q[3];
rz(0.01649012907608413) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.10990634847529168) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.036357267579083426) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.11301807107090742) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.07667271962284164) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.037121676133668534) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.026963950116416266) q[3];
cx q[2],q[3];
rx(-0.23590643236453526) q[0];
rz(0.015376519771615482) q[0];
rx(0.011452996611760942) q[1];
rz(-0.10533503616657537) q[1];
rx(0.04532503396198596) q[2];
rz(-0.09618028631433059) q[2];
rx(-0.20183516167442) q[3];
rz(-0.02117214691985989) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.05579469583193258) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.05371225638047239) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.006447015869551496) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.0397466504920099) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.009511918322780841) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.017106523852863875) q[3];
cx q[2],q[3];
rx(-0.16830310622870662) q[0];
rz(-0.02392942079428676) q[0];
rx(0.015360509643627748) q[1];
rz(-0.14450690704249022) q[1];
rx(0.03887665848732919) q[2];
rz(-0.18691636730497302) q[2];
rx(-0.2480990667913008) q[3];
rz(-0.024624845153511366) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.07432315616181372) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.06472899787964528) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.0009992781312633107) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.07878932237359454) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.01907552981930948) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.06202816593801306) q[3];
cx q[2],q[3];
rx(-0.1399814318170927) q[0];
rz(0.0608967819814326) q[0];
rx(-0.0521836350575571) q[1];
rz(-0.09212713840645954) q[1];
rx(-0.026495529726385487) q[2];
rz(-0.16048498118945576) q[2];
rx(-0.18705040425785358) q[3];
rz(-0.02183704476194647) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.06486350343224942) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.07349946638614796) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.05401409080502201) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.09459860345737) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.032400205842568974) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.021000217095582183) q[3];
cx q[2],q[3];
rx(-0.16908227532116907) q[0];
rz(0.006605471389262666) q[0];
rx(0.005997730472557172) q[1];
rz(-0.1365791381222216) q[1];
rx(-0.0029267652980801574) q[2];
rz(-0.15271822514161623) q[2];
rx(-0.2188090987525305) q[3];
rz(0.05933692307565783) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.08341202583017912) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1377215498123813) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.06338807890912182) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.15073366034730196) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.04226828684700106) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.0014260130847810369) q[3];
cx q[2],q[3];
rx(-0.12487674936170515) q[0];
rz(0.009378391219393075) q[0];
rx(-0.04697345261908337) q[1];
rz(-0.1266035730592114) q[1];
rx(-0.03299027271013678) q[2];
rz(-0.12906997355879882) q[2];
rx(-0.14192189217101775) q[3];
rz(0.02377092902309767) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.16765327327594917) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.10292517177947429) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.07330056888751199) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.15765772030104477) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.0074807365302565305) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.029183474181795618) q[3];
cx q[2],q[3];
rx(-0.1393448992182411) q[0];
rz(-0.026931303590709152) q[0];
rx(0.010003909069418677) q[1];
rz(-0.11140470957684372) q[1];
rx(0.024300013466623055) q[2];
rz(-0.18253236229725336) q[2];
rx(-0.1995533626032131) q[3];
rz(0.012753134980201973) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.154570936330404) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.10441090797202228) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.030387590597747774) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1154906223916014) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.024199534956491634) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.019580095115041593) q[3];
cx q[2],q[3];
rx(-0.12324678507842862) q[0];
rz(0.004039222579967214) q[0];
rx(0.04646770988227808) q[1];
rz(-0.05685322763232936) q[1];
rx(-0.02678714803827999) q[2];
rz(-0.18253147176696344) q[2];
rx(-0.1945967700523572) q[3];
rz(-0.02841033201411372) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.13770378150929066) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.05560523295499706) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.016376293614422107) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.16994651174143816) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.07562444732518271) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.033283147685046116) q[3];
cx q[2],q[3];
rx(-0.17220414864902034) q[0];
rz(-0.062610473110821) q[0];
rx(0.01788117775879892) q[1];
rz(-0.1080071159449459) q[1];
rx(-0.015582256244226812) q[2];
rz(-0.1431643217094647) q[2];
rx(-0.2123639732108741) q[3];
rz(-0.05960349259656618) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.13759924346705588) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.06257912624898494) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.06255041084930911) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.12028844173952356) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.023726557054488596) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.05195192165199424) q[3];
cx q[2],q[3];
rx(-0.11812631942697331) q[0];
rz(-0.005078468806732875) q[0];
rx(-0.04321329481648271) q[1];
rz(-0.017024380773068455) q[1];
rx(-0.06387543862514494) q[2];
rz(-0.13608696270864232) q[2];
rx(-0.18676108587595486) q[3];
rz(-0.08687435167641792) q[3];