OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0300454899056496) q[2];
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
rz(-0.031654456724766494) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.08427098947069689) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.07115427065207704) q[3];
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
rz(-0.1632897284265459) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.04619422991369315) q[3];
cx q[2],q[3];
rx(-0.11729114144415692) q[0];
rz(-0.09266543136589916) q[0];
rx(-0.060095565109726565) q[1];
rz(-0.045408105967027865) q[1];
rx(-0.03299072537365947) q[2];
rz(-0.09394081911661241) q[2];
rx(-0.10407525883459337) q[3];
rz(-0.02882317999669419) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0697442814702753) q[2];
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
rz(0.021595392115614023) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.13504825509086377) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.0022744801428050826) q[3];
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
rz(-0.0949524705042097) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.14240108177261745) q[3];
cx q[2],q[3];
rx(-0.1877349264139613) q[0];
rz(-0.09797621966748078) q[0];
rx(-0.10630272032737217) q[1];
rz(-0.16748076232031658) q[1];
rx(-0.06924936660848854) q[2];
rz(-0.11004997781402903) q[2];
rx(-0.0915423236564984) q[3];
rz(-0.10081038404640953) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.09484138409307181) q[2];
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
rz(-0.029742334035649468) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.10137258094595401) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.14765000928534505) q[3];
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
rz(0.053710277432371493) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.08205319212672209) q[3];
cx q[2],q[3];
rx(-0.2432434748460498) q[0];
rz(-0.10416322424001899) q[0];
rx(-0.03573644506677964) q[1];
rz(-0.15225044175493668) q[1];
rx(0.03293491380860132) q[2];
rz(-0.16957822164861955) q[2];
rx(-0.176268781327088) q[3];
rz(-0.046482205238377) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.13654951677240876) q[2];
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
rz(-0.002975684654584567) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.007250807648435936) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.14640977939806105) q[3];
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
rz(0.09609024616434417) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.1535269197634536) q[3];
cx q[2],q[3];
rx(-0.30696596242335217) q[0];
rz(-0.025152694394447787) q[0];
rx(-0.12828945928720953) q[1];
rz(-0.15660757804982334) q[1];
rx(0.04341915969900483) q[2];
rz(-0.19522164350290253) q[2];
rx(-0.20177409106931796) q[3];
rz(-0.025246966838192976) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.17967146132411477) q[2];
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
rz(0.03252324137245361) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.027884136270468488) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1626649028853943) q[3];
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
rz(0.1750321356380069) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.1774637068420564) q[3];
cx q[2],q[3];
rx(-0.32080261011511657) q[0];
rz(-0.08008419620458504) q[0];
rx(-0.08283644656404042) q[1];
rz(-0.18240826371966423) q[1];
rx(0.06549269539667166) q[2];
rz(-0.1204527396633958) q[2];
rx(-0.17387436897496808) q[3];
rz(-0.054442817322294355) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.24955334775470958) q[2];
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
rz(0.012417560491575656) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.022790209699797708) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.18272589404631362) q[3];
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
rz(0.12082321972977697) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.25423592118541855) q[3];
cx q[2],q[3];
rx(-0.3262106726835662) q[0];
rz(-0.034293244668818036) q[0];
rx(-0.14559499972755713) q[1];
rz(-0.18575347873753129) q[1];
rx(-0.05502596637527198) q[2];
rz(-0.13278485189414543) q[2];
rx(-0.28004488399403743) q[3];
rz(0.0020630290957156594) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.2245486147248308) q[2];
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
rz(0.016810064754445176) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.004765852976921941) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.15150977359202786) q[3];
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
rz(0.1288331863881581) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.1951398801688915) q[3];
cx q[2],q[3];
rx(-0.31109823026990546) q[0];
rz(0.0011995809859089196) q[0];
rx(-0.16624009700093284) q[1];
rz(-0.19094769208323528) q[1];
rx(-0.04663974571282146) q[2];
rz(-0.15208999316807867) q[2];
rx(-0.3185086691111538) q[3];
rz(0.03622298421414938) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.2803226904654084) q[2];
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
rz(0.0899083948321187) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.03253891294077201) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.12975782911112538) q[3];
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
rz(-0.032861086151316374) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.08039333873189103) q[3];
cx q[2],q[3];
rx(-0.2756642264774044) q[0];
rz(0.05876954590165683) q[0];
rx(-0.020085181618727632) q[1];
rz(-0.13756835059859768) q[1];
rx(0.006879726428127606) q[2];
rz(-0.19005594096822867) q[2];
rx(-0.3258518959093624) q[3];
rz(0.028546375083532666) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.22514482072595518) q[2];
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
rz(0.03227759532788942) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.02504330302490487) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.006617625507322583) q[3];
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
rz(-0.1306881387446646) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.03648298253987854) q[3];
cx q[2],q[3];
rx(-0.3220704297838168) q[0];
rz(0.0856975225440947) q[0];
rx(0.015123434675299299) q[1];
rz(-0.22395997142423613) q[1];
rx(0.05919686685381918) q[2];
rz(-0.17005240818372655) q[2];
rx(-0.3206167488173692) q[3];
rz(-0.03310781811742948) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.22794859694024447) q[2];
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
rz(-0.09250211157184832) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.019213473977406803) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.0341415882050935) q[3];
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
rz(-0.2083564068677403) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.04395743440518965) q[3];
cx q[2],q[3];
rx(-0.22507238156622242) q[0];
rz(0.04448024277433318) q[0];
rx(-0.0700880008602078) q[1];
rz(-0.17093344125516619) q[1];
rx(-0.03631426038751155) q[2];
rz(-0.24123834894816606) q[2];
rx(-0.3592050614775569) q[3];
rz(-0.049731354368831714) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.16534275619693248) q[2];
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
rz(-0.2246778345294376) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.0009552411511628456) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.12714750055641785) q[3];
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
rz(-0.19393352423086418) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.002907125103203023) q[3];
cx q[2],q[3];
rx(-0.21139074205901603) q[0];
rz(0.02319830933071612) q[0];
rx(-0.04497880925894537) q[1];
rz(-0.20412182106494586) q[1];
rx(-0.011259472647843957) q[2];
rz(-0.2053092758759431) q[2];
rx(-0.2881374046406573) q[3];
rz(-0.03953487567555882) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.17368226922567834) q[2];
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
rz(-0.19912450688949027) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.05701054614665943) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.08475715164170594) q[3];
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
rz(-0.15032534688559085) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.0105125006545247) q[3];
cx q[2],q[3];
rx(-0.22130174485960452) q[0];
rz(-0.02943901544518889) q[0];
rx(-0.0318222700791738) q[1];
rz(-0.12734098818558423) q[1];
rx(0.026434931559316237) q[2];
rz(-0.17877686241373114) q[2];
rx(-0.3349911249286281) q[3];
rz(0.010892819060410534) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.14717009006061602) q[2];
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
rz(-0.21843965422157388) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.09538180357375675) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1323995916206026) q[3];
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
rz(-0.08345931433550086) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.052441814842992686) q[3];
cx q[2],q[3];
rx(-0.1728593775926651) q[0];
rz(0.0490060373129293) q[0];
rx(-0.027847075754462346) q[1];
rz(-0.11285474678893598) q[1];
rx(0.0010754655693712574) q[2];
rz(-0.07646653595726353) q[2];
rx(-0.3203487578762124) q[3];
rz(0.0005603875764461123) q[3];