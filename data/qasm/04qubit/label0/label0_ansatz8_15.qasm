OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-0.9234031487240758) q[0];
ry(-2.371604454055902) q[1];
cx q[0],q[1];
ry(2.1813600697456352) q[0];
ry(-0.639660284884105) q[1];
cx q[0],q[1];
ry(-2.0504169195781756) q[2];
ry(0.21063883359886937) q[3];
cx q[2],q[3];
ry(2.617231050338226) q[2];
ry(-0.6440364194436459) q[3];
cx q[2],q[3];
ry(-2.974554113482166) q[0];
ry(2.2947434043983197) q[2];
cx q[0],q[2];
ry(-2.559445225088482) q[0];
ry(1.250360437165885) q[2];
cx q[0],q[2];
ry(-1.0772110937744994) q[1];
ry(0.15890539724022545) q[3];
cx q[1],q[3];
ry(-1.6451140558313568) q[1];
ry(2.0853150978699917) q[3];
cx q[1],q[3];
ry(-1.5231339752813493) q[0];
ry(-1.0921347374959105) q[1];
cx q[0],q[1];
ry(0.6067797693091812) q[0];
ry(-2.8712545293371368) q[1];
cx q[0],q[1];
ry(-1.3512791394760666) q[2];
ry(-2.965581174647191) q[3];
cx q[2],q[3];
ry(1.207059244587338) q[2];
ry(0.5995899968479858) q[3];
cx q[2],q[3];
ry(1.3336055401929294) q[0];
ry(-2.2412471721865312) q[2];
cx q[0],q[2];
ry(-2.6883792459743034) q[0];
ry(0.4862967528033124) q[2];
cx q[0],q[2];
ry(2.9381080938348085) q[1];
ry(-1.8681641724437048) q[3];
cx q[1],q[3];
ry(-1.8168618467693038) q[1];
ry(-0.26056577144287285) q[3];
cx q[1],q[3];
ry(-0.9944917423954934) q[0];
ry(-2.873404747008847) q[1];
cx q[0],q[1];
ry(-0.441340205977685) q[0];
ry(-0.4639285551366914) q[1];
cx q[0],q[1];
ry(-2.7349332312572523) q[2];
ry(0.016432467908781057) q[3];
cx q[2],q[3];
ry(-2.4027972492415346) q[2];
ry(-0.1704329020663073) q[3];
cx q[2],q[3];
ry(2.2042980105722) q[0];
ry(-1.1877028010666386) q[2];
cx q[0],q[2];
ry(-1.9905213482934545) q[0];
ry(-2.6402545325943243) q[2];
cx q[0],q[2];
ry(2.5070970730284303) q[1];
ry(2.1503869502948323) q[3];
cx q[1],q[3];
ry(1.829584809872578) q[1];
ry(-2.0573995290320806) q[3];
cx q[1],q[3];
ry(0.7774817981747588) q[0];
ry(2.0898723159787123) q[1];
cx q[0],q[1];
ry(2.286758748997615) q[0];
ry(-0.9801976519381134) q[1];
cx q[0],q[1];
ry(1.6323394211967583) q[2];
ry(-0.5756546637906768) q[3];
cx q[2],q[3];
ry(2.0581644524746006) q[2];
ry(-2.046364132450391) q[3];
cx q[2],q[3];
ry(-3.032623385029037) q[0];
ry(1.230033621299047) q[2];
cx q[0],q[2];
ry(1.0786890999721548) q[0];
ry(2.8702084565253783) q[2];
cx q[0],q[2];
ry(-1.5069559964825556) q[1];
ry(0.08636818994327142) q[3];
cx q[1],q[3];
ry(-3.0054084301356387) q[1];
ry(1.0285716653984576) q[3];
cx q[1],q[3];
ry(-1.5975097700089087) q[0];
ry(-3.0573276179535287) q[1];
cx q[0],q[1];
ry(1.8124742852380644) q[0];
ry(1.449911941848499) q[1];
cx q[0],q[1];
ry(-1.5730870979371774) q[2];
ry(-2.3412749432062565) q[3];
cx q[2],q[3];
ry(1.0174222592859155) q[2];
ry(-2.9559226128659786) q[3];
cx q[2],q[3];
ry(3.0064983615658893) q[0];
ry(-0.38527582703623714) q[2];
cx q[0],q[2];
ry(-2.7771726699370376) q[0];
ry(-1.735824185516604) q[2];
cx q[0],q[2];
ry(-3.0963444764360233) q[1];
ry(-2.3007937738718955) q[3];
cx q[1],q[3];
ry(-2.7091175289010394) q[1];
ry(-0.5543896733726417) q[3];
cx q[1],q[3];
ry(-0.7746349340624129) q[0];
ry(-2.7698188588090336) q[1];
cx q[0],q[1];
ry(-1.6899486060606588) q[0];
ry(0.35967827291625315) q[1];
cx q[0],q[1];
ry(1.6772574180980968) q[2];
ry(-2.7711437221673703) q[3];
cx q[2],q[3];
ry(-1.783472877932746) q[2];
ry(2.6341368346524567) q[3];
cx q[2],q[3];
ry(1.6854342150773711) q[0];
ry(2.2160944062365697) q[2];
cx q[0],q[2];
ry(-2.752771541572013) q[0];
ry(1.9101964009103796) q[2];
cx q[0],q[2];
ry(-1.4201145677678584) q[1];
ry(-0.9797213642695208) q[3];
cx q[1],q[3];
ry(-0.8990378025403274) q[1];
ry(-1.563211037655858) q[3];
cx q[1],q[3];
ry(0.2918709782368557) q[0];
ry(0.6027283185484462) q[1];
cx q[0],q[1];
ry(-0.1151905466994789) q[0];
ry(0.6813177185481534) q[1];
cx q[0],q[1];
ry(-1.7853551918503296) q[2];
ry(-2.2804367583044947) q[3];
cx q[2],q[3];
ry(2.419141046744583) q[2];
ry(-2.944982015088502) q[3];
cx q[2],q[3];
ry(0.16118707077507824) q[0];
ry(0.9801416544242147) q[2];
cx q[0],q[2];
ry(-2.2894463376038536) q[0];
ry(-1.8832157207673887) q[2];
cx q[0],q[2];
ry(-2.43822761934518) q[1];
ry(-0.45609849767667415) q[3];
cx q[1],q[3];
ry(0.5520593167360505) q[1];
ry(2.8476587765540584) q[3];
cx q[1],q[3];
ry(-0.05286143662743559) q[0];
ry(-1.4125407299239758) q[1];
cx q[0],q[1];
ry(-0.28258573496296435) q[0];
ry(-0.4559224817025395) q[1];
cx q[0],q[1];
ry(1.716229186387279) q[2];
ry(-1.9015435820387452) q[3];
cx q[2],q[3];
ry(-0.5449745048394439) q[2];
ry(0.4885476530014623) q[3];
cx q[2],q[3];
ry(2.7657721704140643) q[0];
ry(-2.0909926660993343) q[2];
cx q[0],q[2];
ry(-3.024090143169361) q[0];
ry(-1.1992914613854637) q[2];
cx q[0],q[2];
ry(-0.034923684015447876) q[1];
ry(1.604300162789058) q[3];
cx q[1],q[3];
ry(0.41357169456317117) q[1];
ry(0.6969641010985644) q[3];
cx q[1],q[3];
ry(-0.05344473479876655) q[0];
ry(3.1189130987450193) q[1];
cx q[0],q[1];
ry(3.0144026140308515) q[0];
ry(-3.1132167465685336) q[1];
cx q[0],q[1];
ry(2.158412156706782) q[2];
ry(0.16607824624826217) q[3];
cx q[2],q[3];
ry(-1.8461163259231377) q[2];
ry(0.9609765544581368) q[3];
cx q[2],q[3];
ry(2.323236256848394) q[0];
ry(-0.6859513518948406) q[2];
cx q[0],q[2];
ry(-1.9911952360787346) q[0];
ry(0.15200927570632392) q[2];
cx q[0],q[2];
ry(1.4107238264635518) q[1];
ry(2.6314283142817656) q[3];
cx q[1],q[3];
ry(2.6230806489781795) q[1];
ry(-2.086242116168184) q[3];
cx q[1],q[3];
ry(0.3922562291138634) q[0];
ry(-1.146628697960599) q[1];
cx q[0],q[1];
ry(-0.08295857954690752) q[0];
ry(-0.9574319773865199) q[1];
cx q[0],q[1];
ry(1.9171943915223748) q[2];
ry(-1.6805256755526847) q[3];
cx q[2],q[3];
ry(2.9621774563557866) q[2];
ry(2.7109506737336218) q[3];
cx q[2],q[3];
ry(1.3580067499508741) q[0];
ry(-0.9628855137309674) q[2];
cx q[0],q[2];
ry(-2.731305573155399) q[0];
ry(-0.4584496400866071) q[2];
cx q[0],q[2];
ry(2.266905699687811) q[1];
ry(-0.9027050107165224) q[3];
cx q[1],q[3];
ry(-0.9617830768183169) q[1];
ry(2.0082833070143824) q[3];
cx q[1],q[3];
ry(2.3337005351204008) q[0];
ry(2.7493650140866444) q[1];
cx q[0],q[1];
ry(0.3922782974440402) q[0];
ry(-0.08146565834252631) q[1];
cx q[0],q[1];
ry(3.1016641163707512) q[2];
ry(0.2082338392802665) q[3];
cx q[2],q[3];
ry(0.15945543972297307) q[2];
ry(2.3170795636971038) q[3];
cx q[2],q[3];
ry(0.7353215388503704) q[0];
ry(-1.2882170148961252) q[2];
cx q[0],q[2];
ry(-2.3105914538696526) q[0];
ry(-2.498666475885356) q[2];
cx q[0],q[2];
ry(-0.539541763697418) q[1];
ry(-2.370279307864177) q[3];
cx q[1],q[3];
ry(0.03215430316554891) q[1];
ry(-1.1025997276766608) q[3];
cx q[1],q[3];
ry(1.2130906134827584) q[0];
ry(1.3403970616055976) q[1];
cx q[0],q[1];
ry(2.9446716808548152) q[0];
ry(2.2377719612315476) q[1];
cx q[0],q[1];
ry(0.36910670704846815) q[2];
ry(2.45179332728329) q[3];
cx q[2],q[3];
ry(-0.529484553571376) q[2];
ry(-1.691225844829816) q[3];
cx q[2],q[3];
ry(-2.029675478655464) q[0];
ry(-0.11323719802222332) q[2];
cx q[0],q[2];
ry(0.9966928551113341) q[0];
ry(-0.49345023293741586) q[2];
cx q[0],q[2];
ry(1.793613330490969) q[1];
ry(-3.0269624619641107) q[3];
cx q[1],q[3];
ry(-1.201243376522097) q[1];
ry(2.206848954670445) q[3];
cx q[1],q[3];
ry(-1.244116294995476) q[0];
ry(2.9044802741532867) q[1];
cx q[0],q[1];
ry(1.0181788645448429) q[0];
ry(-1.5759035121353389) q[1];
cx q[0],q[1];
ry(-0.8934251355786328) q[2];
ry(1.9024353263259837) q[3];
cx q[2],q[3];
ry(2.802548661250869) q[2];
ry(-2.1564275581474805) q[3];
cx q[2],q[3];
ry(0.9562355292093033) q[0];
ry(-2.1807657798777207) q[2];
cx q[0],q[2];
ry(-0.4753015432200476) q[0];
ry(2.8921112866632814) q[2];
cx q[0],q[2];
ry(1.7909004412317673) q[1];
ry(-0.5322880134596799) q[3];
cx q[1],q[3];
ry(-2.567493754807593) q[1];
ry(0.9948260116770654) q[3];
cx q[1],q[3];
ry(1.343728022756685) q[0];
ry(-2.837441174131286) q[1];
cx q[0],q[1];
ry(-2.248229214818066) q[0];
ry(-1.250227918689847) q[1];
cx q[0],q[1];
ry(1.3641499839251294) q[2];
ry(0.5649917277926157) q[3];
cx q[2],q[3];
ry(1.4253496978844336) q[2];
ry(-2.033523820952419) q[3];
cx q[2],q[3];
ry(-0.23359891338655905) q[0];
ry(-2.3253091752076513) q[2];
cx q[0],q[2];
ry(-2.050314165019711) q[0];
ry(2.462333513681456) q[2];
cx q[0],q[2];
ry(-2.844519516329643) q[1];
ry(-0.11718407430450295) q[3];
cx q[1],q[3];
ry(0.3770712886112923) q[1];
ry(-1.3384273256314456) q[3];
cx q[1],q[3];
ry(-0.3139903280443654) q[0];
ry(-2.425657008755129) q[1];
cx q[0],q[1];
ry(1.283221768450293) q[0];
ry(1.9862546273198225) q[1];
cx q[0],q[1];
ry(-0.3614238371469003) q[2];
ry(2.0438612539823637) q[3];
cx q[2],q[3];
ry(-1.3401249289733868) q[2];
ry(1.2939489952938095) q[3];
cx q[2],q[3];
ry(2.896512342609716) q[0];
ry(-0.7426916751693343) q[2];
cx q[0],q[2];
ry(-2.9133057319567914) q[0];
ry(2.2299200132665606) q[2];
cx q[0],q[2];
ry(1.7736833251681148) q[1];
ry(-0.5308558763412612) q[3];
cx q[1],q[3];
ry(1.8655687641157996) q[1];
ry(-2.578843206662789) q[3];
cx q[1],q[3];
ry(2.2263714565690003) q[0];
ry(-2.15489416752996) q[1];
cx q[0],q[1];
ry(-0.8740194736173734) q[0];
ry(0.04120753802479253) q[1];
cx q[0],q[1];
ry(2.630725763292082) q[2];
ry(-2.5531216869907687) q[3];
cx q[2],q[3];
ry(0.1572385332636861) q[2];
ry(1.7239075617326156) q[3];
cx q[2],q[3];
ry(-0.6722148856588516) q[0];
ry(-2.7228103702378474) q[2];
cx q[0],q[2];
ry(2.237392193671911) q[0];
ry(-0.21417132765080255) q[2];
cx q[0],q[2];
ry(-2.0219609346976952) q[1];
ry(2.1012782768952176) q[3];
cx q[1],q[3];
ry(-1.7647619137021258) q[1];
ry(-1.6521849443324854) q[3];
cx q[1],q[3];
ry(-2.1724242887510936) q[0];
ry(-0.285073326333273) q[1];
cx q[0],q[1];
ry(-2.838484462767035) q[0];
ry(2.562328525093104) q[1];
cx q[0],q[1];
ry(1.8677689874978007) q[2];
ry(0.24900622072013512) q[3];
cx q[2],q[3];
ry(2.1780731916865896) q[2];
ry(-0.5982041327498315) q[3];
cx q[2],q[3];
ry(-0.7884732795639255) q[0];
ry(0.6871361584343019) q[2];
cx q[0],q[2];
ry(-0.5484406608719252) q[0];
ry(-0.8975018796689369) q[2];
cx q[0],q[2];
ry(-2.7793174682053134) q[1];
ry(-3.0998734320667074) q[3];
cx q[1],q[3];
ry(2.201323594990509) q[1];
ry(2.9551266813473265) q[3];
cx q[1],q[3];
ry(0.4333702642086766) q[0];
ry(-0.5358683451707488) q[1];
cx q[0],q[1];
ry(1.8871821212292756) q[0];
ry(0.5548512068288574) q[1];
cx q[0],q[1];
ry(0.24619843562615937) q[2];
ry(-2.861218639552856) q[3];
cx q[2],q[3];
ry(0.6203188382677851) q[2];
ry(-1.360471377761483) q[3];
cx q[2],q[3];
ry(2.844144598725032) q[0];
ry(2.004258358673896) q[2];
cx q[0],q[2];
ry(-2.759159525093541) q[0];
ry(-2.474703040814638) q[2];
cx q[0],q[2];
ry(-0.6310181098448019) q[1];
ry(0.055506560767703335) q[3];
cx q[1],q[3];
ry(-3.0108022236944745) q[1];
ry(-1.889440614646257) q[3];
cx q[1],q[3];
ry(-3.002150260666496) q[0];
ry(1.9900226415285767) q[1];
ry(1.64050909601358) q[2];
ry(-1.666166922499805) q[3];