OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-0.33720236746743115) q[0];
ry(0.12019463655931914) q[1];
cx q[0],q[1];
ry(1.6561940359587135) q[0];
ry(-1.4523555837055935) q[1];
cx q[0],q[1];
ry(2.911179504327637) q[2];
ry(-0.34205684133333136) q[3];
cx q[2],q[3];
ry(0.4676033602372538) q[2];
ry(2.61333687596611) q[3];
cx q[2],q[3];
ry(-1.8240703977384065) q[4];
ry(0.5247709830411891) q[5];
cx q[4],q[5];
ry(0.02972453356093042) q[4];
ry(2.469041379442701) q[5];
cx q[4],q[5];
ry(-2.4480388108394577) q[6];
ry(-1.553602434641089) q[7];
cx q[6],q[7];
ry(0.9646047885365553) q[6];
ry(-2.753146377771211) q[7];
cx q[6],q[7];
ry(-1.836743410991078) q[8];
ry(1.2895796526753542) q[9];
cx q[8],q[9];
ry(-2.039809594347343) q[8];
ry(2.783147595038477) q[9];
cx q[8],q[9];
ry(1.4764763738511397) q[10];
ry(2.1227524799324433) q[11];
cx q[10],q[11];
ry(-0.8428563178839523) q[10];
ry(2.8300746432241515) q[11];
cx q[10],q[11];
ry(-0.1694532367073167) q[0];
ry(-1.7025759106745006) q[2];
cx q[0],q[2];
ry(-2.3729790666043833) q[0];
ry(0.31113391861327705) q[2];
cx q[0],q[2];
ry(0.06711203168606188) q[2];
ry(2.304638799740045) q[4];
cx q[2],q[4];
ry(1.790960305142666) q[2];
ry(1.3707569089756473) q[4];
cx q[2],q[4];
ry(-2.743598584262884) q[4];
ry(-1.406383255688694) q[6];
cx q[4],q[6];
ry(-2.5621095861181797) q[4];
ry(-2.5685009922623236) q[6];
cx q[4],q[6];
ry(-1.021988939853641) q[6];
ry(2.724921032978068) q[8];
cx q[6],q[8];
ry(6.501604888776068e-05) q[6];
ry(3.141506913033547) q[8];
cx q[6],q[8];
ry(-2.4164607991398754) q[8];
ry(2.2953019358704223) q[10];
cx q[8],q[10];
ry(2.786969864019553) q[8];
ry(-3.0658163227112936) q[10];
cx q[8],q[10];
ry(-0.25643334577320953) q[1];
ry(0.000651855829444159) q[3];
cx q[1],q[3];
ry(2.4279555768324395) q[1];
ry(0.7942123914693276) q[3];
cx q[1],q[3];
ry(-0.46483495267042085) q[3];
ry(-1.4144388332924782) q[5];
cx q[3],q[5];
ry(2.3334501668735923) q[3];
ry(1.9770666589199646) q[5];
cx q[3],q[5];
ry(1.1388433636347368) q[5];
ry(1.0000031710449102) q[7];
cx q[5],q[7];
ry(-0.5084625140725612) q[5];
ry(1.8736012443440977) q[7];
cx q[5],q[7];
ry(-1.8767677481155924) q[7];
ry(-0.38874928309095047) q[9];
cx q[7],q[9];
ry(-3.080331784337581) q[7];
ry(-1.3382672213889087e-05) q[9];
cx q[7],q[9];
ry(0.7741120735243893) q[9];
ry(-0.3744501707139021) q[11];
cx q[9],q[11];
ry(-2.785342837642988) q[9];
ry(-2.32976286381347) q[11];
cx q[9],q[11];
ry(2.017744337694965) q[0];
ry(-0.0019901964522067842) q[1];
cx q[0],q[1];
ry(2.6705209597730604) q[0];
ry(2.287361637089411) q[1];
cx q[0],q[1];
ry(-2.7561322651081044) q[2];
ry(0.31228650593689644) q[3];
cx q[2],q[3];
ry(2.8223219709171636) q[2];
ry(1.3899456970898694) q[3];
cx q[2],q[3];
ry(-2.372505733547849) q[4];
ry(3.0544991242328683) q[5];
cx q[4],q[5];
ry(-0.33400744214575084) q[4];
ry(1.6247514326459145) q[5];
cx q[4],q[5];
ry(-2.7030659085881217) q[6];
ry(-1.1594933474108355) q[7];
cx q[6],q[7];
ry(3.1393020986895452) q[6];
ry(-2.1552320289969207) q[7];
cx q[6],q[7];
ry(1.5517434900651692) q[8];
ry(-1.2457748912656879) q[9];
cx q[8],q[9];
ry(3.011905991991202) q[8];
ry(-2.791976407966121) q[9];
cx q[8],q[9];
ry(-2.7739203962160452) q[10];
ry(-2.8065508570632622) q[11];
cx q[10],q[11];
ry(1.2785170400000265) q[10];
ry(-2.4935383447130217) q[11];
cx q[10],q[11];
ry(-0.8953198946838512) q[0];
ry(1.9356823298221633) q[2];
cx q[0],q[2];
ry(0.4754301411330264) q[0];
ry(-2.3958692336635568) q[2];
cx q[0],q[2];
ry(-1.3849362211697365) q[2];
ry(-1.4532373993704777) q[4];
cx q[2],q[4];
ry(-0.8047906458106633) q[2];
ry(-2.3374806834132906) q[4];
cx q[2],q[4];
ry(2.5029790658705546) q[4];
ry(2.20441636095277) q[6];
cx q[4],q[6];
ry(1.1964112579652806) q[4];
ry(-2.414243027741541) q[6];
cx q[4],q[6];
ry(2.239294421383608) q[6];
ry(1.5268476201250065) q[8];
cx q[6],q[8];
ry(3.1414616447022583) q[6];
ry(1.7380387160592647e-05) q[8];
cx q[6],q[8];
ry(0.9700391575844156) q[8];
ry(-1.1115766998000913) q[10];
cx q[8],q[10];
ry(-1.9838280594403326) q[8];
ry(-2.551052503249451) q[10];
cx q[8],q[10];
ry(0.7191657957837023) q[1];
ry(-0.4399494795140013) q[3];
cx q[1],q[3];
ry(-1.4438371509561065) q[1];
ry(2.1493652867032624) q[3];
cx q[1],q[3];
ry(-0.17540507594344934) q[3];
ry(3.073010750222815) q[5];
cx q[3],q[5];
ry(1.0997845559676631) q[3];
ry(2.8674413802684056) q[5];
cx q[3],q[5];
ry(2.6678578399491673) q[5];
ry(1.5230099758920987) q[7];
cx q[5],q[7];
ry(0.265399163681705) q[5];
ry(0.8093410820153222) q[7];
cx q[5],q[7];
ry(2.929910951455193) q[7];
ry(0.4119761504517525) q[9];
cx q[7],q[9];
ry(-3.141583120198285) q[7];
ry(3.1414778536200054) q[9];
cx q[7],q[9];
ry(1.8993242384155886) q[9];
ry(1.6147724115616784) q[11];
cx q[9],q[11];
ry(0.20241849083209917) q[9];
ry(-0.13991588151618384) q[11];
cx q[9],q[11];
ry(0.7800166686157887) q[0];
ry(2.7155188777712786) q[1];
cx q[0],q[1];
ry(1.800088483740964) q[0];
ry(-2.2652413283470576) q[1];
cx q[0],q[1];
ry(1.792448517059959) q[2];
ry(2.7202420253751995) q[3];
cx q[2],q[3];
ry(-2.5167843966475716) q[2];
ry(-1.9261668553895008) q[3];
cx q[2],q[3];
ry(-1.174150040587631) q[4];
ry(2.3250590918224163) q[5];
cx q[4],q[5];
ry(-1.9015152251795977) q[4];
ry(3.14041558233314) q[5];
cx q[4],q[5];
ry(2.8938184908031626) q[6];
ry(1.750513677767617) q[7];
cx q[6],q[7];
ry(-0.057772312961402505) q[6];
ry(2.9956354025205476) q[7];
cx q[6],q[7];
ry(-1.1013664057596813) q[8];
ry(0.9474154132049355) q[9];
cx q[8],q[9];
ry(-0.5035799854387708) q[8];
ry(1.9235077984691111) q[9];
cx q[8],q[9];
ry(-2.3426349641413324) q[10];
ry(2.755713139834466) q[11];
cx q[10],q[11];
ry(2.1732893700970184) q[10];
ry(0.5561815356170996) q[11];
cx q[10],q[11];
ry(-1.4994366157911116) q[0];
ry(0.5367464041670553) q[2];
cx q[0],q[2];
ry(-1.2091586500409237) q[0];
ry(-3.0074304937800136) q[2];
cx q[0],q[2];
ry(1.187868993459702) q[2];
ry(2.072575209152406) q[4];
cx q[2],q[4];
ry(2.0402029390011247) q[2];
ry(0.42777846005386166) q[4];
cx q[2],q[4];
ry(-2.22242860978481) q[4];
ry(-0.5683579346746123) q[6];
cx q[4],q[6];
ry(-0.6731090296454623) q[4];
ry(0.26207327593336416) q[6];
cx q[4],q[6];
ry(-1.7419710694050863) q[6];
ry(-0.43785792986956484) q[8];
cx q[6],q[8];
ry(-3.14157251776484) q[6];
ry(-3.1415528088783913) q[8];
cx q[6],q[8];
ry(-1.393136456739355) q[8];
ry(0.3432514710590917) q[10];
cx q[8],q[10];
ry(0.09969966152355392) q[8];
ry(2.22799665245664) q[10];
cx q[8],q[10];
ry(-0.23084617766266913) q[1];
ry(-1.9462528388920497) q[3];
cx q[1],q[3];
ry(-0.7949973686705943) q[1];
ry(-0.5177215641556883) q[3];
cx q[1],q[3];
ry(0.7845690535201015) q[3];
ry(-1.366895473708356) q[5];
cx q[3],q[5];
ry(-3.0351585256838423) q[3];
ry(3.022447511306947) q[5];
cx q[3],q[5];
ry(-2.323110504503591) q[5];
ry(-2.890423270370945) q[7];
cx q[5],q[7];
ry(-2.589606646321233) q[5];
ry(0.3230078342062148) q[7];
cx q[5],q[7];
ry(-1.1797777893637917) q[7];
ry(2.3711226615749714) q[9];
cx q[7],q[9];
ry(3.141580362082916) q[7];
ry(-3.1415214192628906) q[9];
cx q[7],q[9];
ry(-1.0582859273882812) q[9];
ry(-0.9843414732893299) q[11];
cx q[9],q[11];
ry(2.8325808455131654) q[9];
ry(-0.4741468405263287) q[11];
cx q[9],q[11];
ry(1.1545125476527311) q[0];
ry(0.6007222116080664) q[1];
cx q[0],q[1];
ry(0.9239187768201516) q[0];
ry(-2.3322691236732567) q[1];
cx q[0],q[1];
ry(0.7546695749526275) q[2];
ry(-0.08152552349291642) q[3];
cx q[2],q[3];
ry(1.975205691007359) q[2];
ry(1.8938846365299442) q[3];
cx q[2],q[3];
ry(1.5373538601241756) q[4];
ry(-1.184676928679548) q[5];
cx q[4],q[5];
ry(2.4520112486029437) q[4];
ry(-1.3841582789691274) q[5];
cx q[4],q[5];
ry(1.3783678572531626) q[6];
ry(-0.06551728856063258) q[7];
cx q[6],q[7];
ry(-0.8237361328905672) q[6];
ry(3.0748749726891207) q[7];
cx q[6],q[7];
ry(-1.7520836320485862) q[8];
ry(-0.6751265217623352) q[9];
cx q[8],q[9];
ry(-1.409949519297398) q[8];
ry(2.6228330563549878) q[9];
cx q[8],q[9];
ry(-1.082377762219192) q[10];
ry(1.2339962720171114) q[11];
cx q[10],q[11];
ry(-2.658554558766898) q[10];
ry(-2.6086717896651215) q[11];
cx q[10],q[11];
ry(2.8506876126522864) q[0];
ry(1.3199416057787277) q[2];
cx q[0],q[2];
ry(1.960055786189292) q[0];
ry(-1.5301726988837308) q[2];
cx q[0],q[2];
ry(-1.4185764181469178) q[2];
ry(0.4953770891367082) q[4];
cx q[2],q[4];
ry(2.5362261247527575) q[2];
ry(-2.7469411885009882) q[4];
cx q[2],q[4];
ry(-0.057667130381071637) q[4];
ry(1.5930630798846073) q[6];
cx q[4],q[6];
ry(-2.9566344807917107) q[4];
ry(-1.690827971321192) q[6];
cx q[4],q[6];
ry(1.8125063076156653) q[6];
ry(1.6387933959429608) q[8];
cx q[6],q[8];
ry(-0.42459377420215355) q[6];
ry(2.4087986908608094e-05) q[8];
cx q[6],q[8];
ry(-0.7305382322922364) q[8];
ry(-2.440416509626772) q[10];
cx q[8],q[10];
ry(0.9341636045162498) q[8];
ry(3.089334931685733) q[10];
cx q[8],q[10];
ry(0.4508710534172949) q[1];
ry(0.4415372568646516) q[3];
cx q[1],q[3];
ry(-0.6030836008372907) q[1];
ry(0.02680139938560622) q[3];
cx q[1],q[3];
ry(-2.559402968189575) q[3];
ry(-0.16008940662091903) q[5];
cx q[3],q[5];
ry(-0.925545165905726) q[3];
ry(0.5121391114062455) q[5];
cx q[3],q[5];
ry(-2.2469711715929384) q[5];
ry(-0.7088319486197961) q[7];
cx q[5],q[7];
ry(-0.9352947403928306) q[5];
ry(0.40854736138338854) q[7];
cx q[5],q[7];
ry(1.020295481510984) q[7];
ry(2.7818304857937917) q[9];
cx q[7],q[9];
ry(-3.1414912198184006) q[7];
ry(2.4605116524689663e-05) q[9];
cx q[7],q[9];
ry(2.498184185855609) q[9];
ry(2.420452351640785) q[11];
cx q[9],q[11];
ry(-2.3647658879101368) q[9];
ry(-0.4390351784524924) q[11];
cx q[9],q[11];
ry(-1.9615917627061297) q[0];
ry(-2.9539100933714155) q[1];
cx q[0],q[1];
ry(2.1956666689549564) q[0];
ry(0.3332619770061573) q[1];
cx q[0],q[1];
ry(2.0004289245752345) q[2];
ry(0.7187513373628088) q[3];
cx q[2],q[3];
ry(0.9898519285894601) q[2];
ry(1.6557792336289132) q[3];
cx q[2],q[3];
ry(-0.08909165889579551) q[4];
ry(0.08861150647102226) q[5];
cx q[4],q[5];
ry(-0.2041663753700549) q[4];
ry(-2.0263262229018006) q[5];
cx q[4],q[5];
ry(2.270664025630065) q[6];
ry(1.426093477647922) q[7];
cx q[6],q[7];
ry(-1.7267912624229549) q[6];
ry(2.0238059715415893) q[7];
cx q[6],q[7];
ry(-0.28837476999622247) q[8];
ry(0.7408502870637985) q[9];
cx q[8],q[9];
ry(3.0404784185329135) q[8];
ry(0.32508364476202894) q[9];
cx q[8],q[9];
ry(-2.289057089755682) q[10];
ry(-1.2016263813884547) q[11];
cx q[10],q[11];
ry(0.042984608595059426) q[10];
ry(-0.08604399602768531) q[11];
cx q[10],q[11];
ry(-2.786263183232525) q[0];
ry(1.2352589860966043) q[2];
cx q[0],q[2];
ry(-0.6747148230027857) q[0];
ry(0.6943089006553169) q[2];
cx q[0],q[2];
ry(-0.7403426315962734) q[2];
ry(0.43579571663611194) q[4];
cx q[2],q[4];
ry(-2.795920166370959) q[2];
ry(0.5096065842805668) q[4];
cx q[2],q[4];
ry(-0.2319906811109486) q[4];
ry(2.8231864354771865) q[6];
cx q[4],q[6];
ry(3.09483851523845) q[4];
ry(0.6440650940279801) q[6];
cx q[4],q[6];
ry(2.8447893381488023) q[6];
ry(1.6587062190710369) q[8];
cx q[6],q[8];
ry(-4.200783927846336e-05) q[6];
ry(2.9750322359411992e-05) q[8];
cx q[6],q[8];
ry(-1.0440456088485686) q[8];
ry(2.451351916326484) q[10];
cx q[8],q[10];
ry(2.9430488229232186) q[8];
ry(1.1259073594284796) q[10];
cx q[8],q[10];
ry(2.0809077625743795) q[1];
ry(1.5612381347811481) q[3];
cx q[1],q[3];
ry(-1.068960082667692) q[1];
ry(0.9123807122853123) q[3];
cx q[1],q[3];
ry(-0.45979758191761366) q[3];
ry(-0.041649241179548824) q[5];
cx q[3],q[5];
ry(2.1843228221877955) q[3];
ry(2.094887608556127) q[5];
cx q[3],q[5];
ry(-2.2742633426540246) q[5];
ry(2.8687952259241585) q[7];
cx q[5],q[7];
ry(-1.581554875484823) q[5];
ry(-2.7276153420233475) q[7];
cx q[5],q[7];
ry(-1.0346771314194259) q[7];
ry(-1.5397121850189643) q[9];
cx q[7],q[9];
ry(-3.1415059632028726) q[7];
ry(1.321989887871551e-05) q[9];
cx q[7],q[9];
ry(1.488698248123442) q[9];
ry(-0.8477109951470205) q[11];
cx q[9],q[11];
ry(-2.3222954427316664) q[9];
ry(-0.8811211748715407) q[11];
cx q[9],q[11];
ry(-0.5562164692085219) q[0];
ry(-0.39308658839086524) q[1];
cx q[0],q[1];
ry(2.9648334713096203) q[0];
ry(-2.998076535041321) q[1];
cx q[0],q[1];
ry(0.4141385980273332) q[2];
ry(2.098139523108887) q[3];
cx q[2],q[3];
ry(-1.9794433191462497) q[2];
ry(-2.3411539126906438) q[3];
cx q[2],q[3];
ry(-2.510715516503879) q[4];
ry(2.7681477165151884) q[5];
cx q[4],q[5];
ry(0.798254199935399) q[4];
ry(0.9352838712277245) q[5];
cx q[4],q[5];
ry(0.6714739959855569) q[6];
ry(-0.1473596315960375) q[7];
cx q[6],q[7];
ry(1.699209912255219) q[6];
ry(2.7033544043104745) q[7];
cx q[6],q[7];
ry(-1.9132903935455556) q[8];
ry(2.5815986441322387) q[9];
cx q[8],q[9];
ry(1.079984401562628) q[8];
ry(-2.467930522325773) q[9];
cx q[8],q[9];
ry(1.5731167472685117) q[10];
ry(-1.9156325254764393) q[11];
cx q[10],q[11];
ry(1.393227871680777) q[10];
ry(1.978823091485905) q[11];
cx q[10],q[11];
ry(-1.4778926089936426) q[0];
ry(3.1350726822039805) q[2];
cx q[0],q[2];
ry(0.7381754669866991) q[0];
ry(2.842099728928569) q[2];
cx q[0],q[2];
ry(-0.021580986132596677) q[2];
ry(1.2120202689149766) q[4];
cx q[2],q[4];
ry(1.4222744469603792) q[2];
ry(2.572788034527361) q[4];
cx q[2],q[4];
ry(2.3602089985203394) q[4];
ry(1.0135498995198438) q[6];
cx q[4],q[6];
ry(2.1055178728083956) q[4];
ry(2.230085562432265) q[6];
cx q[4],q[6];
ry(-0.2981620590289289) q[6];
ry(1.51808947561707) q[8];
cx q[6],q[8];
ry(-3.141561682372577) q[6];
ry(0.00010681159676551034) q[8];
cx q[6],q[8];
ry(1.5453654434982083) q[8];
ry(2.7546853435273086) q[10];
cx q[8],q[10];
ry(1.300883316941876) q[8];
ry(-2.898630904807078) q[10];
cx q[8],q[10];
ry(-1.177418064805978) q[1];
ry(-2.147784908340957) q[3];
cx q[1],q[3];
ry(2.006882809776765) q[1];
ry(-2.3331089414568598) q[3];
cx q[1],q[3];
ry(1.7656015049655898) q[3];
ry(0.9595289075466163) q[5];
cx q[3],q[5];
ry(-1.3919729063248858) q[3];
ry(0.9763377779736553) q[5];
cx q[3],q[5];
ry(0.439904848395353) q[5];
ry(-0.9474680385559591) q[7];
cx q[5],q[7];
ry(-1.6243915500295485) q[5];
ry(1.8056246900511201) q[7];
cx q[5],q[7];
ry(-1.3713235555452021) q[7];
ry(-3.051076586609784) q[9];
cx q[7],q[9];
ry(-5.950159779649041e-05) q[7];
ry(-3.141590101664014) q[9];
cx q[7],q[9];
ry(3.087435827410846) q[9];
ry(1.9921821086722131) q[11];
cx q[9],q[11];
ry(1.9099799772928279) q[9];
ry(0.009357482645138582) q[11];
cx q[9],q[11];
ry(2.3839704369249706) q[0];
ry(2.4165431058449762) q[1];
cx q[0],q[1];
ry(0.13178461836707175) q[0];
ry(-1.175516262806478) q[1];
cx q[0],q[1];
ry(0.7193618781548371) q[2];
ry(-2.4109467458285962) q[3];
cx q[2],q[3];
ry(-0.4592782567274361) q[2];
ry(-1.6071413047687555) q[3];
cx q[2],q[3];
ry(-2.4356939004009086) q[4];
ry(2.8207946809854594) q[5];
cx q[4],q[5];
ry(2.753650315089859) q[4];
ry(0.9280384964723474) q[5];
cx q[4],q[5];
ry(1.5365604161698263) q[6];
ry(-2.213730220314496) q[7];
cx q[6],q[7];
ry(-2.3548089224606845) q[6];
ry(1.3457485232488033) q[7];
cx q[6],q[7];
ry(2.536443549564255) q[8];
ry(2.0599864675037587) q[9];
cx q[8],q[9];
ry(-0.17267528854576067) q[8];
ry(-2.2296044396715766) q[9];
cx q[8],q[9];
ry(-2.8052802440432205) q[10];
ry(-2.8535675739245785) q[11];
cx q[10],q[11];
ry(-1.2457684170879813) q[10];
ry(-1.3897993100723756) q[11];
cx q[10],q[11];
ry(0.39119943385922995) q[0];
ry(-0.6218572928996471) q[2];
cx q[0],q[2];
ry(-0.5595462143470395) q[0];
ry(-2.0112115557783348) q[2];
cx q[0],q[2];
ry(-1.3200465437264493) q[2];
ry(1.0487585825173789) q[4];
cx q[2],q[4];
ry(-0.03558955766323147) q[2];
ry(-0.9092581705683855) q[4];
cx q[2],q[4];
ry(-0.30135176463771896) q[4];
ry(0.7286539042051539) q[6];
cx q[4],q[6];
ry(0.8832569559901655) q[4];
ry(0.5465972394378245) q[6];
cx q[4],q[6];
ry(-1.6558152572257985) q[6];
ry(-2.195188191240078) q[8];
cx q[6],q[8];
ry(-3.141531346567844) q[6];
ry(3.1415597221059737) q[8];
cx q[6],q[8];
ry(0.07089057238029284) q[8];
ry(-1.4999728084789803) q[10];
cx q[8],q[10];
ry(2.693858320801268) q[8];
ry(2.99495585849703) q[10];
cx q[8],q[10];
ry(-0.6566271644370386) q[1];
ry(2.7099611676332307) q[3];
cx q[1],q[3];
ry(2.012227348145222) q[1];
ry(-0.9003337269893168) q[3];
cx q[1],q[3];
ry(2.704132522590247) q[3];
ry(0.991768422419241) q[5];
cx q[3],q[5];
ry(0.46118150352166626) q[3];
ry(2.8949714726781903) q[5];
cx q[3],q[5];
ry(2.827198902835102) q[5];
ry(-1.3315871721554648) q[7];
cx q[5],q[7];
ry(1.950856334217777) q[5];
ry(0.9328981175029486) q[7];
cx q[5],q[7];
ry(2.0239713566852906) q[7];
ry(2.5275412875298846) q[9];
cx q[7],q[9];
ry(-0.00010803675646808841) q[7];
ry(3.1415549020214897) q[9];
cx q[7],q[9];
ry(3.1130950881066304) q[9];
ry(1.7849759291889025) q[11];
cx q[9],q[11];
ry(2.559980753538301) q[9];
ry(3.0988328917032066) q[11];
cx q[9],q[11];
ry(1.925496921691538) q[0];
ry(-3.107686796595978) q[1];
cx q[0],q[1];
ry(0.7834781705421969) q[0];
ry(-1.0680205952086534) q[1];
cx q[0],q[1];
ry(1.0532544239490398) q[2];
ry(-0.47171899255617783) q[3];
cx q[2],q[3];
ry(-2.7102461459660856) q[2];
ry(-0.1870537163869712) q[3];
cx q[2],q[3];
ry(-2.3066522534049625) q[4];
ry(-2.685488852437299) q[5];
cx q[4],q[5];
ry(-3.1041889161328924) q[4];
ry(-0.729289604031793) q[5];
cx q[4],q[5];
ry(0.7897502716298425) q[6];
ry(-1.9713406025173965) q[7];
cx q[6],q[7];
ry(0.8480348936038586) q[6];
ry(2.1349852926401995) q[7];
cx q[6],q[7];
ry(-1.5122940351814516) q[8];
ry(1.584405553969905) q[9];
cx q[8],q[9];
ry(-1.3423414986888158) q[8];
ry(2.625509245354066) q[9];
cx q[8],q[9];
ry(-2.7992910732739915) q[10];
ry(-1.4652763763244878) q[11];
cx q[10],q[11];
ry(-2.143911490676788) q[10];
ry(2.787256386249237) q[11];
cx q[10],q[11];
ry(0.9070959760732729) q[0];
ry(-2.936624320854919) q[2];
cx q[0],q[2];
ry(-2.790278385879399) q[0];
ry(-2.279499543168469) q[2];
cx q[0],q[2];
ry(-2.534728259690925) q[2];
ry(1.8181811745606369) q[4];
cx q[2],q[4];
ry(1.1083697960803844) q[2];
ry(-2.5899416294224302) q[4];
cx q[2],q[4];
ry(1.3996428244542083) q[4];
ry(-3.1244977047771556) q[6];
cx q[4],q[6];
ry(2.564332810195691) q[4];
ry(0.31877078527630154) q[6];
cx q[4],q[6];
ry(2.064762761861657) q[6];
ry(-1.5626428602969238) q[8];
cx q[6],q[8];
ry(-0.07732279965367848) q[6];
ry(3.1366876608948617) q[8];
cx q[6],q[8];
ry(-1.831297124793046) q[8];
ry(-0.5763315991061047) q[10];
cx q[8],q[10];
ry(-3.138488353754523) q[8];
ry(-0.018949743841883304) q[10];
cx q[8],q[10];
ry(0.8069787287165453) q[1];
ry(0.5344818525696844) q[3];
cx q[1],q[3];
ry(1.9492711396719706) q[1];
ry(0.36114361212209284) q[3];
cx q[1],q[3];
ry(-0.608569182567412) q[3];
ry(3.0397902637491736) q[5];
cx q[3],q[5];
ry(-3.063438427864606) q[3];
ry(-1.4621277668001156) q[5];
cx q[3],q[5];
ry(-1.3563997021414846) q[5];
ry(0.8271380375431603) q[7];
cx q[5],q[7];
ry(-0.13096418779006225) q[5];
ry(3.0301876766169697) q[7];
cx q[5],q[7];
ry(1.332419650310932) q[7];
ry(-2.264297774396102) q[9];
cx q[7],q[9];
ry(-2.4553777467016697) q[7];
ry(0.014847598101892956) q[9];
cx q[7],q[9];
ry(-0.6869746399024915) q[9];
ry(1.5292592863704044) q[11];
cx q[9],q[11];
ry(0.00021517653671213567) q[9];
ry(-0.00013715309775139698) q[11];
cx q[9],q[11];
ry(0.2916663279519551) q[0];
ry(-2.2086757231727763) q[1];
cx q[0],q[1];
ry(0.9731339197933897) q[0];
ry(-2.7530354033342284) q[1];
cx q[0],q[1];
ry(-1.7442678380296588) q[2];
ry(3.083431174050241) q[3];
cx q[2],q[3];
ry(-2.7259352283111618) q[2];
ry(-2.6457709138823113) q[3];
cx q[2],q[3];
ry(-3.0386329040751163) q[4];
ry(3.0051783548356705) q[5];
cx q[4],q[5];
ry(0.968169710983668) q[4];
ry(-2.1001609067511304) q[5];
cx q[4],q[5];
ry(-0.7337069737946855) q[6];
ry(1.6259808608282258) q[7];
cx q[6],q[7];
ry(-1.851573666993704) q[6];
ry(0.11281774323400118) q[7];
cx q[6],q[7];
ry(1.9319639345143924) q[8];
ry(1.3884228886384617) q[9];
cx q[8],q[9];
ry(0.43628876635254876) q[8];
ry(3.0788203982552433) q[9];
cx q[8],q[9];
ry(0.9996504622481645) q[10];
ry(-2.1418110979410105) q[11];
cx q[10],q[11];
ry(-2.420575473844557) q[10];
ry(-2.846023397017052) q[11];
cx q[10],q[11];
ry(0.946294643549931) q[0];
ry(-1.8815327987237094) q[2];
cx q[0],q[2];
ry(2.712075021923112) q[0];
ry(1.6080439916904918) q[2];
cx q[0],q[2];
ry(2.653243800631529) q[2];
ry(2.388964099151072) q[4];
cx q[2],q[4];
ry(1.2417085422363305) q[2];
ry(2.215153624297244) q[4];
cx q[2],q[4];
ry(3.0847375455891766) q[4];
ry(-2.94360623997694) q[6];
cx q[4],q[6];
ry(0.015061105912530515) q[4];
ry(3.107384841947886) q[6];
cx q[4],q[6];
ry(1.2293086363554124) q[6];
ry(2.0669383344071557) q[8];
cx q[6],q[8];
ry(0.42044415334146445) q[6];
ry(3.1226949139317486) q[8];
cx q[6],q[8];
ry(2.719753718380069) q[8];
ry(0.5963885847982111) q[10];
cx q[8],q[10];
ry(3.1397642015953107) q[8];
ry(3.141198741970574) q[10];
cx q[8],q[10];
ry(0.9480447333935267) q[1];
ry(1.8273313361198351) q[3];
cx q[1],q[3];
ry(3.113462719677573) q[1];
ry(0.535832934286866) q[3];
cx q[1],q[3];
ry(3.091577013662709) q[3];
ry(1.2001397537212188) q[5];
cx q[3],q[5];
ry(3.0605346838800234) q[3];
ry(-0.21987087210468562) q[5];
cx q[3],q[5];
ry(-2.6192523750508414) q[5];
ry(1.370402813331585) q[7];
cx q[5],q[7];
ry(-0.015001871182949255) q[5];
ry(2.945320336275764) q[7];
cx q[5],q[7];
ry(-2.7518585701733285) q[7];
ry(2.405214436704051) q[9];
cx q[7],q[9];
ry(-1.3425797016930892) q[7];
ry(-3.0953255047232298) q[9];
cx q[7],q[9];
ry(-0.8324322517302041) q[9];
ry(-2.8143453022809335) q[11];
cx q[9],q[11];
ry(0.0038899146320103373) q[9];
ry(0.002774700889255577) q[11];
cx q[9],q[11];
ry(-0.7494786303544939) q[0];
ry(0.8693599639996926) q[1];
cx q[0],q[1];
ry(1.8970837045031497) q[0];
ry(-2.3981946774167135) q[1];
cx q[0],q[1];
ry(1.4324083897269764) q[2];
ry(-1.721937633483478) q[3];
cx q[2],q[3];
ry(1.3526637235745966) q[2];
ry(-1.633019741274596) q[3];
cx q[2],q[3];
ry(-0.4912601590896095) q[4];
ry(-3.1242953878897635) q[5];
cx q[4],q[5];
ry(-1.5496862086590513) q[4];
ry(1.6186587126700978) q[5];
cx q[4],q[5];
ry(1.1930751302034504) q[6];
ry(-1.0713579441941252) q[7];
cx q[6],q[7];
ry(-2.318845814029613) q[6];
ry(-2.1742553303039855) q[7];
cx q[6],q[7];
ry(-2.4028050847856255) q[8];
ry(0.9067166019208752) q[9];
cx q[8],q[9];
ry(0.29408227674835374) q[8];
ry(1.527508016797484) q[9];
cx q[8],q[9];
ry(-2.1488133211554783) q[10];
ry(-0.16878830617329577) q[11];
cx q[10],q[11];
ry(0.560510573296205) q[10];
ry(-2.1278398758844066) q[11];
cx q[10],q[11];
ry(-1.2363212007471984) q[0];
ry(-0.7816167975002001) q[2];
cx q[0],q[2];
ry(0.5857629380566829) q[0];
ry(1.5217193096810622) q[2];
cx q[0],q[2];
ry(-0.0902172266114416) q[2];
ry(-0.31231291304768294) q[4];
cx q[2],q[4];
ry(0.004425750351666569) q[2];
ry(0.0022466592511740657) q[4];
cx q[2],q[4];
ry(-0.10245378470318567) q[4];
ry(1.8753819956021713) q[6];
cx q[4],q[6];
ry(-1.059315738198292) q[4];
ry(2.7165013096657495) q[6];
cx q[4],q[6];
ry(1.4481684581965037) q[6];
ry(-2.0720845180330505) q[8];
cx q[6],q[8];
ry(-1.7016306226772864) q[6];
ry(0.5461560170827582) q[8];
cx q[6],q[8];
ry(-0.6298698381867247) q[8];
ry(0.4552398882012947) q[10];
cx q[8],q[10];
ry(-0.009696928985982822) q[8];
ry(-3.1413993840753585) q[10];
cx q[8],q[10];
ry(-0.3530481904742535) q[1];
ry(1.9302352280018915) q[3];
cx q[1],q[3];
ry(1.6916167666474209) q[1];
ry(-2.5546127443711315) q[3];
cx q[1],q[3];
ry(-1.0982023236628535) q[3];
ry(1.8979443595547791) q[5];
cx q[3],q[5];
ry(-3.0999970646087895) q[3];
ry(-3.0865877895676355) q[5];
cx q[3],q[5];
ry(-1.8339357012339788) q[5];
ry(-1.780690506474734) q[7];
cx q[5],q[7];
ry(-0.16870062114430884) q[5];
ry(0.0014349597221068322) q[7];
cx q[5],q[7];
ry(1.2334833207947407) q[7];
ry(-1.2115308775002724) q[9];
cx q[7],q[9];
ry(0.4203224242428254) q[7];
ry(0.06812060834802924) q[9];
cx q[7],q[9];
ry(1.0208519017109712) q[9];
ry(-1.6921994192499747) q[11];
cx q[9],q[11];
ry(-0.02272128321883629) q[9];
ry(0.0053133595609464734) q[11];
cx q[9],q[11];
ry(1.3925955087469903) q[0];
ry(0.16794225561944565) q[1];
cx q[0],q[1];
ry(0.8010702771519149) q[0];
ry(-0.3327334818495993) q[1];
cx q[0],q[1];
ry(-1.8735515142026902) q[2];
ry(-0.2999742975860796) q[3];
cx q[2],q[3];
ry(2.210141832854262) q[2];
ry(3.1257522044188195) q[3];
cx q[2],q[3];
ry(3.043311202640879) q[4];
ry(1.990242037350268) q[5];
cx q[4],q[5];
ry(0.07370291296219862) q[4];
ry(-0.36133355049471483) q[5];
cx q[4],q[5];
ry(-0.011686864056014379) q[6];
ry(0.2096716359785047) q[7];
cx q[6],q[7];
ry(3.093622163180845) q[6];
ry(-0.62337375442132) q[7];
cx q[6],q[7];
ry(-0.3838607267813492) q[8];
ry(1.0443618998315163) q[9];
cx q[8],q[9];
ry(-1.4709843919978338) q[8];
ry(1.7421752790730345) q[9];
cx q[8],q[9];
ry(1.7513212085784928) q[10];
ry(0.30671335380489406) q[11];
cx q[10],q[11];
ry(-0.553930932844465) q[10];
ry(-1.7509346820263287) q[11];
cx q[10],q[11];
ry(0.2230691799843942) q[0];
ry(-2.7795094452719264) q[2];
cx q[0],q[2];
ry(0.5770075020724801) q[0];
ry(2.210097108118015) q[2];
cx q[0],q[2];
ry(0.7205738666868604) q[2];
ry(-0.8031115899654555) q[4];
cx q[2],q[4];
ry(0.004227700817258477) q[2];
ry(-0.0003590067230438976) q[4];
cx q[2],q[4];
ry(-0.4960992870189367) q[4];
ry(-0.5686735898227857) q[6];
cx q[4],q[6];
ry(-0.7425184387625001) q[4];
ry(2.858905369663085) q[6];
cx q[4],q[6];
ry(-0.0944832658485173) q[6];
ry(-0.08486861830233661) q[8];
cx q[6],q[8];
ry(-0.006020562692364473) q[6];
ry(3.1359374895970618) q[8];
cx q[6],q[8];
ry(-0.9517815492747382) q[8];
ry(-2.879172846714313) q[10];
cx q[8],q[10];
ry(-0.00233705894752991) q[8];
ry(3.1399265304663095) q[10];
cx q[8],q[10];
ry(-1.8065130909628253) q[1];
ry(3.042915317001293) q[3];
cx q[1],q[3];
ry(-3.1310973344281816) q[1];
ry(2.0647030505667168) q[3];
cx q[1],q[3];
ry(0.7717591398006879) q[3];
ry(2.6624549732533525) q[5];
cx q[3],q[5];
ry(3.1263230673455884) q[3];
ry(-0.3056581620631622) q[5];
cx q[3],q[5];
ry(-0.8241366771103332) q[5];
ry(3.0990067183577987) q[7];
cx q[5],q[7];
ry(-0.03484390407698858) q[5];
ry(2.9387010355701837) q[7];
cx q[5],q[7];
ry(1.5403057866508487) q[7];
ry(-0.9124881838863201) q[9];
cx q[7],q[9];
ry(2.1756517269556985) q[7];
ry(0.007278561683397112) q[9];
cx q[7],q[9];
ry(2.0834264172972787) q[9];
ry(-0.5130752235506915) q[11];
cx q[9],q[11];
ry(-2.9948387401095515) q[9];
ry(0.08617798895558426) q[11];
cx q[9],q[11];
ry(-2.558195825035028) q[0];
ry(-1.4676194916031502) q[1];
cx q[0],q[1];
ry(-2.264749454596619) q[0];
ry(-0.13121410583302007) q[1];
cx q[0],q[1];
ry(0.08239920457702167) q[2];
ry(-1.4330294001918331) q[3];
cx q[2],q[3];
ry(0.9346957043058505) q[2];
ry(-0.569237481029375) q[3];
cx q[2],q[3];
ry(1.6031079325169009) q[4];
ry(2.302905999910166) q[5];
cx q[4],q[5];
ry(0.3194416083861653) q[4];
ry(2.957007641197723) q[5];
cx q[4],q[5];
ry(0.49428643380216464) q[6];
ry(0.22561815804408422) q[7];
cx q[6],q[7];
ry(3.1014204204207276) q[6];
ry(0.23300387256067917) q[7];
cx q[6],q[7];
ry(-1.6727974537810013) q[8];
ry(-0.5036907058292925) q[9];
cx q[8],q[9];
ry(-3.0081062364497106) q[8];
ry(1.0187928944121456) q[9];
cx q[8],q[9];
ry(-0.7005263347144964) q[10];
ry(1.353755812901705) q[11];
cx q[10],q[11];
ry(0.29484950162845386) q[10];
ry(-0.41958267671783184) q[11];
cx q[10],q[11];
ry(-1.3250109767414964) q[0];
ry(1.0228500808664647) q[2];
cx q[0],q[2];
ry(-0.4657343105442573) q[0];
ry(1.1029074526122167) q[2];
cx q[0],q[2];
ry(0.8948108524724052) q[2];
ry(3.0512783998741657) q[4];
cx q[2],q[4];
ry(3.140829025994477) q[2];
ry(-0.009728311973228206) q[4];
cx q[2],q[4];
ry(-1.150583171699435) q[4];
ry(-0.05218172806090875) q[6];
cx q[4],q[6];
ry(-2.3374608911518666) q[4];
ry(0.2359441740637012) q[6];
cx q[4],q[6];
ry(-2.672092242806414) q[6];
ry(-2.5827801320462282) q[8];
cx q[6],q[8];
ry(-0.12182151347520173) q[6];
ry(-2.987298888569087) q[8];
cx q[6],q[8];
ry(0.38723678600441325) q[8];
ry(1.1256397206992563) q[10];
cx q[8],q[10];
ry(-0.013273449454834517) q[8];
ry(-0.06803605633084474) q[10];
cx q[8],q[10];
ry(2.232877690506924) q[1];
ry(-2.689398157265381) q[3];
cx q[1],q[3];
ry(0.6288994883992958) q[1];
ry(-2.3188431079789793) q[3];
cx q[1],q[3];
ry(-2.0005548065673286) q[3];
ry(1.2129813505911289) q[5];
cx q[3],q[5];
ry(3.1370997211786924) q[3];
ry(0.0054902760198117335) q[5];
cx q[3],q[5];
ry(1.5992221264986972) q[5];
ry(0.13925111278611935) q[7];
cx q[5],q[7];
ry(-3.1410261481765187) q[5];
ry(-3.07416052714319) q[7];
cx q[5],q[7];
ry(-1.4878291611915286) q[7];
ry(-0.2990596435255224) q[9];
cx q[7],q[9];
ry(3.0275647506160563) q[7];
ry(0.006347517631004141) q[9];
cx q[7],q[9];
ry(-1.2083385959468071) q[9];
ry(0.5628100582671367) q[11];
cx q[9],q[11];
ry(-2.976382255664536) q[9];
ry(-0.02759434448916289) q[11];
cx q[9],q[11];
ry(-0.576512404878993) q[0];
ry(1.4123484002850115) q[1];
cx q[0],q[1];
ry(1.3434227465332853) q[0];
ry(-0.28626667029334685) q[1];
cx q[0],q[1];
ry(-1.737310432278828) q[2];
ry(1.5832231533257854) q[3];
cx q[2],q[3];
ry(-1.1446495360279503) q[2];
ry(-1.5767969087756313) q[3];
cx q[2],q[3];
ry(-0.43726858635377924) q[4];
ry(-3.117476888817367) q[5];
cx q[4],q[5];
ry(0.9149355983342018) q[4];
ry(-3.0166886771652037) q[5];
cx q[4],q[5];
ry(-1.7349080044319072) q[6];
ry(-1.6084618902381544) q[7];
cx q[6],q[7];
ry(-0.02627898129193916) q[6];
ry(-1.5771008060852354) q[7];
cx q[6],q[7];
ry(-0.46240530134994007) q[8];
ry(-0.11887307428323926) q[9];
cx q[8],q[9];
ry(1.8146836027176798) q[8];
ry(1.3965530281040621) q[9];
cx q[8],q[9];
ry(-2.1388001262385163) q[10];
ry(1.5721501613563564) q[11];
cx q[10],q[11];
ry(-2.0471471337224347) q[10];
ry(1.1730809602158692) q[11];
cx q[10],q[11];
ry(-0.2611600089308972) q[0];
ry(-0.9087910501072862) q[2];
cx q[0],q[2];
ry(0.16884445088056044) q[0];
ry(1.262426417973429) q[2];
cx q[0],q[2];
ry(-2.6809271498976566) q[2];
ry(-1.0206476458575422) q[4];
cx q[2],q[4];
ry(0.02211830745803435) q[2];
ry(-0.049609378235344614) q[4];
cx q[2],q[4];
ry(-2.2948047582071) q[4];
ry(1.6277939406619133) q[6];
cx q[4],q[6];
ry(-0.03866208185263531) q[4];
ry(-3.1160822499128837) q[6];
cx q[4],q[6];
ry(-2.3095280457679976) q[6];
ry(0.5131010700494637) q[8];
cx q[6],q[8];
ry(0.008120444763503265) q[6];
ry(1.2900887831756365) q[8];
cx q[6],q[8];
ry(-0.10097584620815248) q[8];
ry(2.67586328168187) q[10];
cx q[8],q[10];
ry(-0.26068389594896235) q[8];
ry(3.1405334806753773) q[10];
cx q[8],q[10];
ry(0.33661993977295346) q[1];
ry(1.2680194126662532) q[3];
cx q[1],q[3];
ry(-1.1262651788374791) q[1];
ry(0.2517259113614075) q[3];
cx q[1],q[3];
ry(1.0826883985426776) q[3];
ry(2.6570740012459932) q[5];
cx q[3],q[5];
ry(-0.0007997142043583191) q[3];
ry(3.1364525905443) q[5];
cx q[3],q[5];
ry(-0.7046765695134232) q[5];
ry(1.98414720998505) q[7];
cx q[5],q[7];
ry(-3.134828356020793) q[5];
ry(2.983748980962874) q[7];
cx q[5],q[7];
ry(-0.7879655831308999) q[7];
ry(0.9136044214371923) q[9];
cx q[7],q[9];
ry(1.4966996404325594) q[7];
ry(3.095487291713922) q[9];
cx q[7],q[9];
ry(2.219457192949429) q[9];
ry(1.1282155886657028) q[11];
cx q[9],q[11];
ry(0.0032312401070555556) q[9];
ry(3.138480415111853) q[11];
cx q[9],q[11];
ry(1.6116247610736478) q[0];
ry(-0.06187712111929238) q[1];
cx q[0],q[1];
ry(1.3919764065464886) q[0];
ry(-2.3552547299600506) q[1];
cx q[0],q[1];
ry(2.3094133352274593) q[2];
ry(1.170888385409195) q[3];
cx q[2],q[3];
ry(0.07801814504149718) q[2];
ry(3.0273349776188465) q[3];
cx q[2],q[3];
ry(-0.7361988232711836) q[4];
ry(1.0277694956881602) q[5];
cx q[4],q[5];
ry(-0.6814235996538631) q[4];
ry(1.621074982805841) q[5];
cx q[4],q[5];
ry(-0.02143245704054352) q[6];
ry(2.639148294353973) q[7];
cx q[6],q[7];
ry(-3.090034259538186) q[6];
ry(1.808564432184225) q[7];
cx q[6],q[7];
ry(2.849907532695081) q[8];
ry(-0.7424225106760343) q[9];
cx q[8],q[9];
ry(1.4513353674681877) q[8];
ry(3.1408760128778894) q[9];
cx q[8],q[9];
ry(0.8286156765242136) q[10];
ry(-0.8210620170970033) q[11];
cx q[10],q[11];
ry(-2.96372948130235) q[10];
ry(0.8511529464503285) q[11];
cx q[10],q[11];
ry(2.3311255547166128) q[0];
ry(2.4149572640202375) q[2];
cx q[0],q[2];
ry(-3.0388200807391605) q[0];
ry(-3.1038630300937955) q[2];
cx q[0],q[2];
ry(-1.6000732114519678) q[2];
ry(-1.7419790632527041) q[4];
cx q[2],q[4];
ry(-3.1024425322381655) q[2];
ry(0.0024704787111522464) q[4];
cx q[2],q[4];
ry(3.100557846955564) q[4];
ry(-1.603098985242597) q[6];
cx q[4],q[6];
ry(-0.01480883972971192) q[4];
ry(3.1407546840912164) q[6];
cx q[4],q[6];
ry(-1.5259881362857854) q[6];
ry(-0.8437537911934907) q[8];
cx q[6],q[8];
ry(-3.099685282761156) q[6];
ry(-1.882063794569626) q[8];
cx q[6],q[8];
ry(-0.30653026851385956) q[8];
ry(1.8470450780922931) q[10];
cx q[8],q[10];
ry(-2.9551067933272104) q[8];
ry(-3.1344882494685353) q[10];
cx q[8],q[10];
ry(-1.1970439558039576) q[1];
ry(0.056867997601219855) q[3];
cx q[1],q[3];
ry(1.2891775528643086) q[1];
ry(-3.1114985782843907) q[3];
cx q[1],q[3];
ry(1.092430433562445) q[3];
ry(1.8542765893399444) q[5];
cx q[3],q[5];
ry(-3.1032239344905337) q[3];
ry(-3.0858676033878285) q[5];
cx q[3],q[5];
ry(2.774136961356904) q[5];
ry(1.7082566057619317) q[7];
cx q[5],q[7];
ry(-0.0015795375343745008) q[5];
ry(-3.123295954177431) q[7];
cx q[5],q[7];
ry(1.5552044499617406) q[7];
ry(2.8076717114633776) q[9];
cx q[7],q[9];
ry(-1.5550339956670622) q[7];
ry(-3.103666623811208) q[9];
cx q[7],q[9];
ry(-2.4987093826186317) q[9];
ry(3.0606622437400897) q[11];
cx q[9],q[11];
ry(-0.06622924360787272) q[9];
ry(-3.1019141245836455) q[11];
cx q[9],q[11];
ry(-1.7585915194398865) q[0];
ry(-1.7428923756005128) q[1];
ry(0.5653696693540854) q[2];
ry(0.7636638896046941) q[3];
ry(0.7461974537487972) q[4];
ry(-2.063123886508695) q[5];
ry(-0.4511325866771978) q[6];
ry(0.40811951249064987) q[7];
ry(-0.3855458285406276) q[8];
ry(2.220331012153681) q[9];
ry(2.575413079221905) q[10];
ry(0.050068181790416943) q[11];