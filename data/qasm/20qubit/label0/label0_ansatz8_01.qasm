OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(2.9328775707034027) q[0];
ry(-2.0726941820885516) q[1];
cx q[0],q[1];
ry(0.1745609313673156) q[0];
ry(0.05990510411772952) q[1];
cx q[0],q[1];
ry(-1.9960416866319435) q[2];
ry(-1.092620895067986) q[3];
cx q[2],q[3];
ry(-1.9837911887056088) q[2];
ry(1.733886399422231) q[3];
cx q[2],q[3];
ry(3.1278612155283128) q[4];
ry(2.6472799115457435) q[5];
cx q[4],q[5];
ry(3.1246430355053962) q[4];
ry(0.6233612095018284) q[5];
cx q[4],q[5];
ry(-0.18719852220033914) q[6];
ry(-2.2056483910167017) q[7];
cx q[6],q[7];
ry(-2.5186125838190194) q[6];
ry(-0.49270016423053153) q[7];
cx q[6],q[7];
ry(0.8503347987589531) q[8];
ry(-1.4062839682129626) q[9];
cx q[8],q[9];
ry(0.8647892865046201) q[8];
ry(-1.5076085843973157) q[9];
cx q[8],q[9];
ry(-2.8954487104802022) q[10];
ry(2.1025916511871463) q[11];
cx q[10],q[11];
ry(0.4705783764709313) q[10];
ry(-2.695011650782174) q[11];
cx q[10],q[11];
ry(-3.138844680106209) q[12];
ry(-2.6717427675871726) q[13];
cx q[12],q[13];
ry(-0.4711187152905403) q[12];
ry(2.027748834187352) q[13];
cx q[12],q[13];
ry(2.692764832411015) q[14];
ry(-1.5418395725443999) q[15];
cx q[14],q[15];
ry(-0.460483014534513) q[14];
ry(0.8172041713268486) q[15];
cx q[14],q[15];
ry(2.8086050814043153) q[16];
ry(-0.28374319348522636) q[17];
cx q[16],q[17];
ry(-2.864731832233881) q[16];
ry(-2.545595236933108) q[17];
cx q[16],q[17];
ry(2.739939631367903) q[18];
ry(1.7683740364230527) q[19];
cx q[18],q[19];
ry(-1.1406461931043932) q[18];
ry(0.1795089916471302) q[19];
cx q[18],q[19];
ry(1.1363292881090734) q[0];
ry(1.490975257686593) q[2];
cx q[0],q[2];
ry(2.2861139837591478) q[0];
ry(1.0344701991462282) q[2];
cx q[0],q[2];
ry(-1.8725939385425185) q[2];
ry(1.6156756430587604) q[4];
cx q[2],q[4];
ry(-2.0487240240327727) q[2];
ry(-0.018118978088420334) q[4];
cx q[2],q[4];
ry(-2.923771325715755) q[4];
ry(-2.005113132667567) q[6];
cx q[4],q[6];
ry(0.08455388621349583) q[4];
ry(0.02449197678030135) q[6];
cx q[4],q[6];
ry(2.805940611119494) q[6];
ry(-1.4447904441906685) q[8];
cx q[6],q[8];
ry(2.5998930677468985) q[6];
ry(-3.1413358880335025) q[8];
cx q[6],q[8];
ry(-3.1023785492390497) q[8];
ry(0.6482185540240394) q[10];
cx q[8],q[10];
ry(-1.6423791284272315) q[8];
ry(1.4976223023675237) q[10];
cx q[8],q[10];
ry(-0.21126655059336757) q[10];
ry(2.912375053179278) q[12];
cx q[10],q[12];
ry(-3.074544576574156) q[10];
ry(2.9753587875463987) q[12];
cx q[10],q[12];
ry(-0.8530282273890655) q[12];
ry(-2.725878624441169) q[14];
cx q[12],q[14];
ry(-0.0002730784354293952) q[12];
ry(-0.030431763361892714) q[14];
cx q[12],q[14];
ry(-2.1846172487863313) q[14];
ry(-2.7671227701759933) q[16];
cx q[14],q[16];
ry(2.857146699699774) q[14];
ry(2.9576433272030274) q[16];
cx q[14],q[16];
ry(2.7948112902860682) q[16];
ry(-2.6854520257800973) q[18];
cx q[16],q[18];
ry(-1.0774905451677446) q[16];
ry(1.1547634791602075) q[18];
cx q[16],q[18];
ry(-1.1191792463572645) q[1];
ry(-0.2867555427694715) q[3];
cx q[1],q[3];
ry(-0.5570446702324109) q[1];
ry(-1.7312484204962084) q[3];
cx q[1],q[3];
ry(-0.16866851067854835) q[3];
ry(-2.3412032696566834) q[5];
cx q[3],q[5];
ry(0.18461351234799114) q[3];
ry(0.052109102209661806) q[5];
cx q[3],q[5];
ry(-0.39832569929157724) q[5];
ry(-1.9144752543452037) q[7];
cx q[5],q[7];
ry(0.00398355522468119) q[5];
ry(1.966590207005289) q[7];
cx q[5],q[7];
ry(1.3200051057527014) q[7];
ry(1.2540455166668538) q[9];
cx q[7],q[9];
ry(-0.025805339045751552) q[7];
ry(3.1403183373312347) q[9];
cx q[7],q[9];
ry(-0.09718098353047377) q[9];
ry(0.22172218747164962) q[11];
cx q[9],q[11];
ry(-2.5422136309666405) q[9];
ry(0.18744422331506708) q[11];
cx q[9],q[11];
ry(2.6124000734754693) q[11];
ry(1.7104913826112973) q[13];
cx q[11],q[13];
ry(-0.0025436997873044607) q[11];
ry(0.004602096557570157) q[13];
cx q[11],q[13];
ry(-1.6719197303860929) q[13];
ry(-0.2804689706465032) q[15];
cx q[13],q[15];
ry(-0.00890332310980569) q[13];
ry(0.0077762498814903935) q[15];
cx q[13],q[15];
ry(-1.7111573350140228) q[15];
ry(0.49590298069483485) q[17];
cx q[15],q[17];
ry(1.5754703106034436) q[15];
ry(2.538995825385195) q[17];
cx q[15],q[17];
ry(2.2113830254602256) q[17];
ry(2.290305479238874) q[19];
cx q[17],q[19];
ry(-2.361091399516695) q[17];
ry(2.3442114732694206) q[19];
cx q[17],q[19];
ry(-2.8763917536357124) q[0];
ry(-2.9795293155732416) q[1];
cx q[0],q[1];
ry(2.4429546320735414) q[0];
ry(0.9338491525753687) q[1];
cx q[0],q[1];
ry(0.7402140859455068) q[2];
ry(0.4939832444319545) q[3];
cx q[2],q[3];
ry(1.5704510780236385) q[2];
ry(-1.3685166119217136) q[3];
cx q[2],q[3];
ry(-1.5844396745327005) q[4];
ry(-1.5920702430946623) q[5];
cx q[4],q[5];
ry(2.2100836305452756) q[4];
ry(-0.022904651730057072) q[5];
cx q[4],q[5];
ry(2.8689912917644413) q[6];
ry(-1.92046561113247) q[7];
cx q[6],q[7];
ry(-3.139479019048354) q[6];
ry(0.6207705304743141) q[7];
cx q[6],q[7];
ry(1.4796298818598421) q[8];
ry(-1.087343272102312) q[9];
cx q[8],q[9];
ry(0.5454187402453731) q[8];
ry(-2.5745183007158863) q[9];
cx q[8],q[9];
ry(-0.7942075848059815) q[10];
ry(0.8677665326131266) q[11];
cx q[10],q[11];
ry(2.1035708442921046) q[10];
ry(3.0118561656095912) q[11];
cx q[10],q[11];
ry(-0.46317743521164) q[12];
ry(-0.002732241096214061) q[13];
cx q[12],q[13];
ry(-0.41776917184059487) q[12];
ry(3.131616190739438) q[13];
cx q[12],q[13];
ry(1.4494467200458638) q[14];
ry(-1.4351744238047344) q[15];
cx q[14],q[15];
ry(-3.006310306395787) q[14];
ry(-2.334616186466172) q[15];
cx q[14],q[15];
ry(0.19651205785309134) q[16];
ry(2.414235105275782) q[17];
cx q[16],q[17];
ry(2.818015736301439) q[16];
ry(2.684860167245403) q[17];
cx q[16],q[17];
ry(0.5615010394441136) q[18];
ry(2.6852528778248144) q[19];
cx q[18],q[19];
ry(-0.07357820803511128) q[18];
ry(-1.5092501587971645) q[19];
cx q[18],q[19];
ry(2.948459182236526) q[0];
ry(2.9511264339805785) q[2];
cx q[0],q[2];
ry(0.644363745966337) q[0];
ry(0.7699703914899629) q[2];
cx q[0],q[2];
ry(-0.7128490069653658) q[2];
ry(-2.9485542335829) q[4];
cx q[2],q[4];
ry(-3.1402223357712336) q[2];
ry(3.134403847442483) q[4];
cx q[2],q[4];
ry(-1.860029193938497) q[4];
ry(2.670637886169884) q[6];
cx q[4],q[6];
ry(0.715657320564768) q[4];
ry(-2.777709132790165) q[6];
cx q[4],q[6];
ry(-2.995222557326569) q[6];
ry(-0.9172314922732596) q[8];
cx q[6],q[8];
ry(-3.127134109963278) q[6];
ry(2.693617239148292) q[8];
cx q[6],q[8];
ry(0.13734081310095822) q[8];
ry(-0.046426986283164065) q[10];
cx q[8],q[10];
ry(3.062762210331754) q[8];
ry(0.004266780307916385) q[10];
cx q[8],q[10];
ry(1.9484170421613183) q[10];
ry(2.0967013642732555) q[12];
cx q[10],q[12];
ry(-2.9207266999816626) q[10];
ry(-2.654132656475338) q[12];
cx q[10],q[12];
ry(-1.5335561494061274) q[12];
ry(1.5470226629273434) q[14];
cx q[12],q[14];
ry(-0.040194429622176564) q[12];
ry(-3.1387219208370145) q[14];
cx q[12],q[14];
ry(-0.9519360266567388) q[14];
ry(-2.6190437529140027) q[16];
cx q[14],q[16];
ry(-1.3537261136253917) q[14];
ry(-1.873511971197753) q[16];
cx q[14],q[16];
ry(-2.066659623261354) q[16];
ry(-0.1732042390301567) q[18];
cx q[16],q[18];
ry(-0.000788671554131426) q[16];
ry(-0.010660864169416116) q[18];
cx q[16],q[18];
ry(1.1545300650815062) q[1];
ry(0.16279494269480654) q[3];
cx q[1],q[3];
ry(0.9321447468846616) q[1];
ry(2.763105035580591) q[3];
cx q[1],q[3];
ry(-1.9201485658084538) q[3];
ry(-3.0330213718426102) q[5];
cx q[3],q[5];
ry(0.001204454832480069) q[3];
ry(-3.141493202263678) q[5];
cx q[3],q[5];
ry(-1.67623661000085) q[5];
ry(1.4390220056954266) q[7];
cx q[5],q[7];
ry(3.127258564290689) q[5];
ry(1.9993802413710404) q[7];
cx q[5],q[7];
ry(1.0568522789928902) q[7];
ry(-1.3163189757567377) q[9];
cx q[7],q[9];
ry(0.33180967524077776) q[7];
ry(-2.464530165184265) q[9];
cx q[7],q[9];
ry(2.5955497724237313) q[9];
ry(-0.07345969588075274) q[11];
cx q[9],q[11];
ry(-1.7666923397538046) q[9];
ry(-3.141167510145197) q[11];
cx q[9],q[11];
ry(0.32605322279070836) q[11];
ry(1.7053424416578054) q[13];
cx q[11],q[13];
ry(-0.01564478502721923) q[11];
ry(-0.01747862605561408) q[13];
cx q[11],q[13];
ry(-1.5134614311728898) q[13];
ry(-2.6342202665213925) q[15];
cx q[13],q[15];
ry(3.0364523067415474) q[13];
ry(0.41108128893931867) q[15];
cx q[13],q[15];
ry(-2.3946409366442567) q[15];
ry(-0.34313800993256294) q[17];
cx q[15],q[17];
ry(-1.953530137837087) q[15];
ry(-0.008509342709820622) q[17];
cx q[15],q[17];
ry(2.7465463843567024) q[17];
ry(-0.8141891853680229) q[19];
cx q[17],q[19];
ry(3.133781936126861) q[17];
ry(-0.008707370883045405) q[19];
cx q[17],q[19];
ry(-1.738171663749221) q[0];
ry(2.386689301280294) q[1];
cx q[0],q[1];
ry(0.9948977053198345) q[0];
ry(-0.7160912283052614) q[1];
cx q[0],q[1];
ry(-2.2319541867173793) q[2];
ry(1.2406549431352998) q[3];
cx q[2],q[3];
ry(2.1523884445708337) q[2];
ry(0.12420362465737662) q[3];
cx q[2],q[3];
ry(-2.734744143376168) q[4];
ry(2.1042026250553176) q[5];
cx q[4],q[5];
ry(0.3813968015988968) q[4];
ry(-2.8337951058825337) q[5];
cx q[4],q[5];
ry(-0.09974331878071041) q[6];
ry(1.1265376539711287) q[7];
cx q[6],q[7];
ry(-0.0005665521052785119) q[6];
ry(0.24826983755782894) q[7];
cx q[6],q[7];
ry(-2.270926997668944) q[8];
ry(3.1251155118787093) q[9];
cx q[8],q[9];
ry(-3.1088051559238687) q[8];
ry(0.0057805446301157914) q[9];
cx q[8],q[9];
ry(-2.3515779385041484) q[10];
ry(-0.7006166201760644) q[11];
cx q[10],q[11];
ry(2.9866760567616297) q[10];
ry(0.34844210249728214) q[11];
cx q[10],q[11];
ry(-0.9578395433394054) q[12];
ry(1.4977784958729672) q[13];
cx q[12],q[13];
ry(3.1391198718443616) q[12];
ry(-3.140711951930027) q[13];
cx q[12],q[13];
ry(-0.5975464623320937) q[14];
ry(1.7567497167535826) q[15];
cx q[14],q[15];
ry(3.0767066885955874) q[14];
ry(2.6740358614017112) q[15];
cx q[14],q[15];
ry(0.3854560321388272) q[16];
ry(2.3590738968334497) q[17];
cx q[16],q[17];
ry(2.7041814127128387) q[16];
ry(2.1393619166114153) q[17];
cx q[16],q[17];
ry(0.2802170549596401) q[18];
ry(-0.17533861186650126) q[19];
cx q[18],q[19];
ry(-1.5784001031595694) q[18];
ry(0.8435791879177286) q[19];
cx q[18],q[19];
ry(-2.891236311174023) q[0];
ry(1.9814590563691281) q[2];
cx q[0],q[2];
ry(1.5908853334219597) q[0];
ry(-1.81977581413165) q[2];
cx q[0],q[2];
ry(0.5439446748358717) q[2];
ry(1.7866742459185518) q[4];
cx q[2],q[4];
ry(0.009645204331457779) q[2];
ry(3.1415732057658086) q[4];
cx q[2],q[4];
ry(2.0843625641797074) q[4];
ry(-0.016882846300686174) q[6];
cx q[4],q[6];
ry(0.0025923257679237454) q[4];
ry(-0.03957406791799922) q[6];
cx q[4],q[6];
ry(0.06482197804741752) q[6];
ry(-0.035960451160310124) q[8];
cx q[6],q[8];
ry(-3.1331937079326297) q[6];
ry(0.269680073824306) q[8];
cx q[6],q[8];
ry(-2.3713008796510495) q[8];
ry(1.299955223453695) q[10];
cx q[8],q[10];
ry(2.976657129920797) q[8];
ry(0.08958708769691068) q[10];
cx q[8],q[10];
ry(-2.850615222705214) q[10];
ry(-2.1430476389761433) q[12];
cx q[10],q[12];
ry(0.28754003797471306) q[10];
ry(-2.714872373115731) q[12];
cx q[10],q[12];
ry(-2.7395622250210625) q[12];
ry(-2.5602056229725916) q[14];
cx q[12],q[14];
ry(-0.044398283024918335) q[12];
ry(-2.9342055021787785) q[14];
cx q[12],q[14];
ry(1.544772592691443) q[14];
ry(0.17401561942293428) q[16];
cx q[14],q[16];
ry(-3.1219465866992153) q[14];
ry(-0.14650665646540698) q[16];
cx q[14],q[16];
ry(1.068420867459265) q[16];
ry(0.7666750604758636) q[18];
cx q[16],q[18];
ry(-3.126338288245978) q[16];
ry(3.0108334868473023) q[18];
cx q[16],q[18];
ry(-2.203104147541453) q[1];
ry(1.49820111181904) q[3];
cx q[1],q[3];
ry(-2.0856467247341657) q[1];
ry(1.6326744670846915) q[3];
cx q[1],q[3];
ry(-2.0165166203751594) q[3];
ry(-1.6535136031221922) q[5];
cx q[3],q[5];
ry(2.876896010217414) q[3];
ry(-0.006632157308855469) q[5];
cx q[3],q[5];
ry(-1.2100249397560705) q[5];
ry(1.7846823124502391) q[7];
cx q[5],q[7];
ry(0.0003717695742461652) q[5];
ry(0.0028589352827008696) q[7];
cx q[5],q[7];
ry(0.623088392175779) q[7];
ry(-2.5926360942681903) q[9];
cx q[7],q[9];
ry(2.579770345406739) q[7];
ry(-0.028814600132633927) q[9];
cx q[7],q[9];
ry(-2.5097339130140535) q[9];
ry(-0.8606047147821103) q[11];
cx q[9],q[11];
ry(-2.0521455908061563) q[9];
ry(-0.0009668040851531501) q[11];
cx q[9],q[11];
ry(-0.40317555778756853) q[11];
ry(-2.9157388228082595) q[13];
cx q[11],q[13];
ry(-0.004106990104557535) q[11];
ry(-0.005468799178678374) q[13];
cx q[11],q[13];
ry(1.5533443450366287) q[13];
ry(-1.4016944298516985) q[15];
cx q[13],q[15];
ry(1.4856971079748273) q[13];
ry(1.0873815171290886) q[15];
cx q[13],q[15];
ry(1.679489920150086) q[15];
ry(1.2140488675407983) q[17];
cx q[15],q[17];
ry(1.5833550222603625) q[15];
ry(1.5849502312841688) q[17];
cx q[15],q[17];
ry(-1.6081713518216443) q[17];
ry(1.948770882346099) q[19];
cx q[17],q[19];
ry(3.0145515823762734) q[17];
ry(-2.92164258882738) q[19];
cx q[17],q[19];
ry(0.2428920519732456) q[0];
ry(0.3806226903994281) q[1];
cx q[0],q[1];
ry(-3.01808974780379) q[0];
ry(-0.03556911614879737) q[1];
cx q[0],q[1];
ry(0.7075379578100179) q[2];
ry(-2.877086918415245) q[3];
cx q[2],q[3];
ry(0.24474524480345342) q[2];
ry(2.448154647256948) q[3];
cx q[2],q[3];
ry(1.1816095226469001) q[4];
ry(-2.752412018898445) q[5];
cx q[4],q[5];
ry(-1.656973978241429) q[4];
ry(1.5767568386349495) q[5];
cx q[4],q[5];
ry(0.764801415317096) q[6];
ry(1.3253331143192832) q[7];
cx q[6],q[7];
ry(-0.004009445122100885) q[6];
ry(3.0368621408426804) q[7];
cx q[6],q[7];
ry(2.560139970249795) q[8];
ry(-0.6728423675252877) q[9];
cx q[8],q[9];
ry(3.0972591255311013) q[8];
ry(3.1383989690324916) q[9];
cx q[8],q[9];
ry(-2.5958285468704516) q[10];
ry(1.7690217944713078) q[11];
cx q[10],q[11];
ry(-1.1909972009920178) q[10];
ry(1.717546949015044) q[11];
cx q[10],q[11];
ry(3.1239550701010206) q[12];
ry(-1.6855404191683556) q[13];
cx q[12],q[13];
ry(-2.967369303424108) q[12];
ry(2.691222942557782) q[13];
cx q[12],q[13];
ry(-3.0983360081204587) q[14];
ry(-1.0382605156021232) q[15];
cx q[14],q[15];
ry(0.08615641246266093) q[14];
ry(-2.783417764064959) q[15];
cx q[14],q[15];
ry(0.02388191787861209) q[16];
ry(-0.8968244772838889) q[17];
cx q[16],q[17];
ry(0.003901586176598753) q[16];
ry(0.28359180632252184) q[17];
cx q[16],q[17];
ry(0.5869843534185613) q[18];
ry(-2.36541938497101) q[19];
cx q[18],q[19];
ry(-1.7546812140915318) q[18];
ry(-2.7869556819111345) q[19];
cx q[18],q[19];
ry(1.6773246124672472) q[0];
ry(2.872808867665336) q[2];
cx q[0],q[2];
ry(-3.115151947938923) q[0];
ry(2.514829464519262) q[2];
cx q[0],q[2];
ry(1.7586232664255528) q[2];
ry(0.6457870620510229) q[4];
cx q[2],q[4];
ry(0.007010611145875565) q[2];
ry(-0.14372901541629535) q[4];
cx q[2],q[4];
ry(0.9177143843014102) q[4];
ry(2.2997014580220454) q[6];
cx q[4],q[6];
ry(-0.02821705045808211) q[4];
ry(3.084761555918351) q[6];
cx q[4],q[6];
ry(0.47945797882176583) q[6];
ry(0.9708102623068217) q[8];
cx q[6],q[8];
ry(-3.0694528339866918) q[6];
ry(0.013076417801515028) q[8];
cx q[6],q[8];
ry(1.6145458615191366) q[8];
ry(-1.5647756277484421) q[10];
cx q[8],q[10];
ry(-3.069971106960088) q[8];
ry(-0.14815920585056147) q[10];
cx q[8],q[10];
ry(1.5225538731591035) q[10];
ry(-2.5620311412121666) q[12];
cx q[10],q[12];
ry(0.00047360052829561723) q[10];
ry(0.00603258587387856) q[12];
cx q[10],q[12];
ry(2.5090223810813885) q[12];
ry(-1.608881686387594) q[14];
cx q[12],q[14];
ry(3.1176454765258157) q[12];
ry(-3.098025813312461) q[14];
cx q[12],q[14];
ry(2.706373951509546) q[14];
ry(2.3783632451196524) q[16];
cx q[14],q[16];
ry(-3.1400870753545598) q[14];
ry(-3.1411477327560187) q[16];
cx q[14],q[16];
ry(-0.7321329848411988) q[16];
ry(-1.2368276491950756) q[18];
cx q[16],q[18];
ry(3.0912050960587427) q[16];
ry(-2.873839802562075) q[18];
cx q[16],q[18];
ry(-1.4834590070839653) q[1];
ry(1.096428525436611) q[3];
cx q[1],q[3];
ry(-0.000885569599418872) q[1];
ry(0.0862003791637882) q[3];
cx q[1],q[3];
ry(-0.6044896967860605) q[3];
ry(-1.6565141345405099) q[5];
cx q[3],q[5];
ry(-2.9312026849781048) q[3];
ry(-0.005065827465578465) q[5];
cx q[3],q[5];
ry(-1.5800583613248103) q[5];
ry(0.03061266592857153) q[7];
cx q[5],q[7];
ry(-0.07632829476601088) q[5];
ry(3.034220856460968) q[7];
cx q[5],q[7];
ry(-0.9386969320652723) q[7];
ry(2.4932284650446945) q[9];
cx q[7],q[9];
ry(3.114942476218863) q[7];
ry(3.0670091211714556) q[9];
cx q[7],q[9];
ry(2.350046532465236) q[9];
ry(0.22889738290089756) q[11];
cx q[9],q[11];
ry(0.12238593590732187) q[9];
ry(0.017569355001778142) q[11];
cx q[9],q[11];
ry(2.039867566533477) q[11];
ry(-2.9016533658752226) q[13];
cx q[11],q[13];
ry(-0.006013070256729721) q[11];
ry(-0.009739170003229656) q[13];
cx q[11],q[13];
ry(0.4770443579374931) q[13];
ry(2.4871617576740968) q[15];
cx q[13],q[15];
ry(3.1156238698067846) q[13];
ry(-3.1406819350718065) q[15];
cx q[13],q[15];
ry(3.09459330723309) q[15];
ry(-0.4981986238613508) q[17];
cx q[15],q[17];
ry(-0.0006281567573884732) q[15];
ry(3.1355540606107404) q[17];
cx q[15],q[17];
ry(-1.836025490099593) q[17];
ry(-0.09599720821926282) q[19];
cx q[17],q[19];
ry(0.13160512883134143) q[17];
ry(3.096859917714358) q[19];
cx q[17],q[19];
ry(2.7830151733934314) q[0];
ry(-0.045825064640923376) q[1];
ry(-0.0025722879329963533) q[2];
ry(-2.083859568469398) q[3];
ry(3.1228593275904473) q[4];
ry(-0.00435397247041891) q[5];
ry(0.982829091440511) q[6];
ry(-2.5291849259075088) q[7];
ry(3.135050655277921) q[8];
ry(-2.8969559183934233) q[9];
ry(0.032740095309713446) q[10];
ry(2.842660957285307) q[11];
ry(-0.04980361841021113) q[12];
ry(-0.8689515573263575) q[13];
ry(-1.9725461375607096) q[14];
ry(-1.4963532051122233) q[15];
ry(-0.0009217268995776217) q[16];
ry(3.0385683425570287) q[17];
ry(-1.4202081058111737) q[18];
ry(1.4007541691727932) q[19];