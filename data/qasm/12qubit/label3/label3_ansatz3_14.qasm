OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-1.4554862555493013) q[0];
rz(-1.412131828118155) q[0];
ry(-1.7898847929502508) q[1];
rz(-1.9053187858651124) q[1];
ry(-2.569569744184316) q[2];
rz(-2.178453165388553) q[2];
ry(0.4335501024471098) q[3];
rz(-0.2109131039562181) q[3];
ry(-1.2219398390170335) q[4];
rz(-1.1185001650519466) q[4];
ry(3.135677786002183) q[5];
rz(1.3116233846256302) q[5];
ry(-3.1352329754279276) q[6];
rz(-0.4576244378140046) q[6];
ry(2.107594134048073) q[7];
rz(0.35769455051906096) q[7];
ry(-0.5415940653189658) q[8];
rz(-2.5127058740144803) q[8];
ry(1.474537484943962) q[9];
rz(-1.8946561199953544) q[9];
ry(-0.6078946004939425) q[10];
rz(2.873832087083144) q[10];
ry(2.0949429137431137) q[11];
rz(2.6250911330978806) q[11];
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
ry(-2.1432476042183417) q[0];
rz(1.6222812157875097) q[0];
ry(-0.40907660895161607) q[1];
rz(-0.09074379038803392) q[1];
ry(-1.6606042577568594) q[2];
rz(-1.594741752413062) q[2];
ry(1.088073176455592) q[3];
rz(2.1149313177640368) q[3];
ry(-2.873678112364357) q[4];
rz(3.018623443522281) q[4];
ry(3.1333434053053) q[5];
rz(0.15504230217346485) q[5];
ry(0.002028945370481594) q[6];
rz(1.792699755044631) q[6];
ry(-0.639848857224349) q[7];
rz(-2.2638874944899303) q[7];
ry(-2.3477188847297654) q[8];
rz(-1.6017315537209136) q[8];
ry(-1.6198771985467646) q[9];
rz(0.4954289313963898) q[9];
ry(1.4111247907636573) q[10];
rz(-0.3196370392962926) q[10];
ry(0.7269728995368036) q[11];
rz(1.5666242954730798) q[11];
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
ry(1.7437531435078792) q[0];
rz(1.6108052187987971) q[0];
ry(2.4290444349141658) q[1];
rz(-3.0667987148869855) q[1];
ry(-1.600440002693075) q[2];
rz(1.250766421819571) q[2];
ry(-1.5994014088769577) q[3];
rz(-0.5282188766165924) q[3];
ry(1.3689863793706545) q[4];
rz(0.8686353144500825) q[4];
ry(3.139894343633741) q[5];
rz(1.8673768907085444) q[5];
ry(0.0027205128816127906) q[6];
rz(-2.656498691948798) q[6];
ry(-1.354673388620108) q[7];
rz(2.7142116645761245) q[7];
ry(-2.025624149473257) q[8];
rz(2.358797469743604) q[8];
ry(-1.2002342591679707) q[9];
rz(-1.2579682603637128) q[9];
ry(-2.179837265297418) q[10];
rz(-1.7628339225430327) q[10];
ry(0.08587651911425898) q[11];
rz(-2.2536663828330834) q[11];
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
ry(-1.783712887952512) q[0];
rz(2.5322780503048086) q[0];
ry(2.306070804971209) q[1];
rz(-2.963277205900037) q[1];
ry(1.3503816217576317) q[2];
rz(-1.6669273923054346) q[2];
ry(-0.23826974310200155) q[3];
rz(-1.168804511051733) q[3];
ry(-3.074177065309424) q[4];
rz(-0.42281872642785334) q[4];
ry(-3.1360518701177864) q[5];
rz(1.146702790701939) q[5];
ry(0.0009122634410010377) q[6];
rz(2.9940632197959567) q[6];
ry(-0.2951197863696852) q[7];
rz(-2.7944411547646424) q[7];
ry(1.3809748111479685) q[8];
rz(-2.8077719056596386) q[8];
ry(1.8374937719869817) q[9];
rz(-2.4330016634210905) q[9];
ry(1.008059172361939) q[10];
rz(-0.6066486532974276) q[10];
ry(2.2790277520589752) q[11];
rz(0.4539681659689237) q[11];
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
ry(-1.7117197112353566) q[0];
rz(-1.514570458681253) q[0];
ry(1.7941916806798706) q[1];
rz(1.1510775962233537) q[1];
ry(2.1127006783617492) q[2];
rz(-0.06360523709466115) q[2];
ry(2.9566318023167684) q[3];
rz(2.353504675081249) q[3];
ry(1.3872515100367009) q[4];
rz(-1.464363841184606) q[4];
ry(-0.00010613314052942968) q[5];
rz(-1.1894278881856346) q[5];
ry(-1.569801443909815) q[6];
rz(2.0544410104313195) q[6];
ry(-2.475452788924266) q[7];
rz(2.6363591565968734) q[7];
ry(-0.5139454232988918) q[8];
rz(0.18649858990281754) q[8];
ry(-1.7887854965151773) q[9];
rz(-2.6754751767615894) q[9];
ry(-1.980922934973492) q[10];
rz(2.1302785434083322) q[10];
ry(-2.212239241246194) q[11];
rz(-1.9638284560345942) q[11];
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
ry(2.081878667083866) q[0];
rz(0.05798533361196653) q[0];
ry(2.2359782525253427) q[1];
rz(3.1085523332126215) q[1];
ry(-1.543996778714778) q[2];
rz(1.2830300937726857) q[2];
ry(0.5445813877440395) q[3];
rz(-1.5745821798957218) q[3];
ry(0.0010374968810697993) q[4];
rz(1.8368995260426593) q[4];
ry(0.010334204040032267) q[5];
rz(2.05595223494167) q[5];
ry(-0.000350743166523948) q[6];
rz(2.6545544680943283) q[6];
ry(1.5717049187071996) q[7];
rz(1.5688543185122956) q[7];
ry(1.5709595037544632) q[8];
rz(-0.8878176307086132) q[8];
ry(-2.7558513168100633) q[9];
rz(2.098286170157463) q[9];
ry(-0.30173687945113625) q[10];
rz(-2.3320320676637687) q[10];
ry(1.601117020823417) q[11];
rz(1.6350677549161092) q[11];
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
ry(2.90400339376216) q[0];
rz(1.332933209402156) q[0];
ry(1.4352523259918442) q[1];
rz(1.43882912497794) q[1];
ry(0.7610578649040689) q[2];
rz(0.048858125083332204) q[2];
ry(-3.0266984733982434) q[3];
rz(1.4015967006031052) q[3];
ry(1.5946217907072775) q[4];
rz(1.5663364234946078) q[4];
ry(0.003770227696157047) q[5];
rz(0.5312513956378044) q[5];
ry(2.535184701216581) q[6];
rz(-2.5337842311605723) q[6];
ry(1.5701915291485022) q[7];
rz(-2.163429650282795) q[7];
ry(-0.00020339644470938217) q[8];
rz(-0.6788629458272833) q[8];
ry(-1.0401568591809103) q[9];
rz(-1.503775527552163) q[9];
ry(-1.5710714444173364) q[10];
rz(3.1344849160816564) q[10];
ry(-1.3419363307467878) q[11];
rz(-0.04138067132492295) q[11];
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
ry(1.8897654297714763) q[0];
rz(0.8611453279239881) q[0];
ry(-1.8292544767638983) q[1];
rz(-0.7222696564429887) q[1];
ry(-1.617054834056277) q[2];
rz(-0.33645946134825344) q[2];
ry(0.8465801810532658) q[3];
rz(-3.0098051676434388) q[3];
ry(-1.5749711342827741) q[4];
rz(1.572007685006783) q[4];
ry(-2.2448935677619785e-05) q[5];
rz(2.755239979811651) q[5];
ry(-3.140755763639051) q[6];
rz(-0.5001360696622061) q[6];
ry(0.5676926884127704) q[7];
rz(2.4025160527297005) q[7];
ry(-1.570881140862686) q[8];
rz(0.3123981514475553) q[8];
ry(1.572469434260039) q[9];
rz(1.569834732822513) q[9];
ry(-2.204132282754691) q[10];
rz(3.1382586848765603) q[10];
ry(1.5741697848190326) q[11];
rz(-2.794970509237271) q[11];
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
ry(2.206729812528236) q[0];
rz(2.7653243015028663) q[0];
ry(1.138027381585843) q[1];
rz(-0.03627835492073624) q[1];
ry(1.569560355137853) q[2];
rz(-1.030551950295722) q[2];
ry(-2.7882111553915903) q[3];
rz(2.5717330978055357) q[3];
ry(1.5715170264888638) q[4];
rz(1.0614035646734314) q[4];
ry(1.5771325452518072) q[5];
rz(3.1384756653139383) q[5];
ry(-3.1412635590024003) q[6];
rz(2.25628904813104) q[6];
ry(-1.6057241138391987) q[7];
rz(3.037454376868361) q[7];
ry(-2.7244024625037184) q[8];
rz(1.4679804286075386) q[8];
ry(1.5737039450336416) q[9];
rz(3.0847930153345144) q[9];
ry(1.5804832835594507) q[10];
rz(-2.0390905155227568) q[10];
ry(0.005826065947307582) q[11];
rz(0.7941520301582434) q[11];
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
ry(-0.0005264434920446703) q[0];
rz(-0.14876499968815404) q[0];
ry(-1.8250453569185234) q[1];
rz(-0.2526557264533249) q[1];
ry(-0.0009302085105877822) q[2];
rz(-0.5515602040858639) q[2];
ry(0.05293704354385813) q[3];
rz(-1.5611286373374558) q[3];
ry(2.910671845442533) q[4];
rz(0.9590552741531653) q[4];
ry(2.8155691951525545) q[5];
rz(1.5676099827766343) q[5];
ry(-3.1379597049324586) q[6];
rz(0.5500520191287865) q[6];
ry(3.1396330145908196) q[7];
rz(1.2416847635552273) q[7];
ry(1.567395151881219) q[8];
rz(-0.005551299965407202) q[8];
ry(0.5339372576064487) q[9];
rz(-0.23420672538161952) q[9];
ry(-0.7548942258757103) q[10];
rz(1.5605776998824377) q[10];
ry(0.005832690131414964) q[11];
rz(0.43633927666912076) q[11];
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
ry(2.3229180479961347) q[0];
rz(2.93669178196401) q[0];
ry(-0.7943572479265751) q[1];
rz(2.076318090199362) q[1];
ry(1.5713028683062067) q[2];
rz(0.841538963961753) q[2];
ry(1.5516039024732928) q[3];
rz(-0.79797903527222) q[3];
ry(-0.00019033282158272604) q[4];
rz(0.613692439502148) q[4];
ry(-1.5521724440848583) q[5];
rz(1.5646364138562827) q[5];
ry(-0.0003995211140424004) q[6];
rz(1.2023908092985653) q[6];
ry(0.011150662660763365) q[7];
rz(2.87192362374927) q[7];
ry(-2.899835671636252) q[8];
rz(1.5608919137225472) q[8];
ry(2.1239393950109204) q[9];
rz(-2.3725015720072347) q[9];
ry(-1.1313179323962599) q[10];
rz(-1.3654237140997265) q[10];
ry(0.9366335790878747) q[11];
rz(2.5968225496185515) q[11];
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
ry(3.141331050232747) q[0];
rz(-0.5513563351542068) q[0];
ry(3.1271205695871434) q[1];
rz(1.6944657527785818) q[1];
ry(0.0002233112382043308) q[2];
rz(-2.4388536725661867) q[2];
ry(3.141226552059691) q[3];
rz(-0.7544541201068862) q[3];
ry(1.5710028491044195) q[4];
rz(1.7563158634099256) q[4];
ry(2.8353522304205008) q[5];
rz(-1.5704404803265648) q[5];
ry(1.5697810376215804) q[6];
rz(0.026790954295797694) q[6];
ry(2.915785437114662) q[7];
rz(-1.5989166818531364) q[7];
ry(-1.3453302316286437) q[8];
rz(-3.089647873801773) q[8];
ry(2.7982585873545696) q[9];
rz(-1.1233071503377952) q[9];
ry(-0.8760731378180981) q[10];
rz(2.536213332862973) q[10];
ry(0.013067737202637096) q[11];
rz(0.5087160897540013) q[11];
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
ry(1.332067976430368) q[0];
rz(2.7983052447608214) q[0];
ry(2.3212609036844056) q[1];
rz(-1.9256606346089034) q[1];
ry(-1.5715370307106993) q[2];
rz(0.1519245255682461) q[2];
ry(1.5926123472021283) q[3];
rz(1.8314800406997644) q[3];
ry(0.0010100835260393737) q[4];
rz(-0.1853862694824126) q[4];
ry(2.909808644159981) q[5];
rz(-2.708648534440408) q[5];
ry(6.781411348615033e-05) q[6];
rz(-1.521429865072145) q[6];
ry(0.20138196020995236) q[7];
rz(-3.103078462109126) q[7];
ry(-1.5725895701399775e-05) q[8];
rz(-3.0778228322904195) q[8];
ry(1.5834161389885386) q[9];
rz(-1.5672264922007146) q[9];
ry(-2.8297387124920843) q[10];
rz(3.096555406743568) q[10];
ry(-0.16461107397872698) q[11];
rz(0.0271374293031444) q[11];
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
ry(-3.140632947499045) q[0];
rz(-1.591569736041848) q[0];
ry(1.5737684966621552) q[1];
rz(-3.0551188652500616) q[1];
ry(0.0002447295610753446) q[2];
rz(1.4188527775943833) q[2];
ry(-3.141592290978445) q[3];
rz(-2.8853396522801207) q[3];
ry(-1.5702174399727515) q[4];
rz(2.9194953060040234) q[4];
ry(3.1405021857398276) q[5];
rz(-0.5029510381231295) q[5];
ry(1.5642189357914447) q[6];
rz(-1.5525790847126908) q[6];
ry(2.864643889299583) q[7];
rz(-2.524861265951979) q[7];
ry(0.0018784888763045922) q[8];
rz(-1.3874372119052298) q[8];
ry(-1.5706815798080926) q[9];
rz(0.005858789656255806) q[9];
ry(-2.1193552949666064) q[10];
rz(-2.289027867583481) q[10];
ry(1.6973551673456946) q[11];
rz(1.5705692096129322) q[11];
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
ry(1.5462989197162296) q[0];
rz(-1.5763855995890006) q[0];
ry(-0.18018649256144761) q[1];
rz(-0.4554049582243138) q[1];
ry(1.5667013816270172) q[2];
rz(-2.8455457613422186) q[2];
ry(1.5579035562523327) q[3];
rz(-1.569177729707232) q[3];
ry(0.0006635797931586751) q[4];
rz(0.5043777485464178) q[4];
ry(0.0004900236042766971) q[5];
rz(2.7745379349416264) q[5];
ry(-1.5690617231553157) q[6];
rz(-0.00040830237043820716) q[6];
ry(-3.1413323046272383) q[7];
rz(0.19291910283815614) q[7];
ry(-3.141589435238142) q[8];
rz(-2.844158270655104) q[8];
ry(1.5687957926465241) q[9];
rz(-1.7926109819664706) q[9];
ry(-1.5707844408900833) q[10];
rz(-3.1380694208881383) q[10];
ry(1.5364993717508648) q[11];
rz(3.1408963545755393) q[11];
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
ry(-1.567978852021386) q[0];
rz(-1.5161592520625755) q[0];
ry(-3.0598302252156864) q[1];
rz(-0.3673079448090446) q[1];
ry(-3.1414814969891016) q[2];
rz(0.17044060518197468) q[2];
ry(1.570170568623208) q[3];
rz(-0.37746366089619077) q[3];
ry(3.139564176415379) q[4];
rz(2.3280533650174946) q[4];
ry(-0.3548838109640548) q[5];
rz(2.6221901222047657) q[5];
ry(-1.5212967671804083) q[6];
rz(-0.00046793264813871366) q[6];
ry(3.139209493687948) q[7];
rz(0.31304793511833595) q[7];
ry(0.22663614809062427) q[8];
rz(-2.7086025146362527) q[8];
ry(0.0023045558332057676) q[9];
rz(-2.9207407686434825) q[9];
ry(-1.552495446070668) q[10];
rz(-1.6484585032216534) q[10];
ry(-1.5720390638254538) q[11];
rz(-0.0025863919306030685) q[11];
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
ry(1.5813483228945018) q[0];
rz(1.3525607017256709) q[0];
ry(-1.569871740496839) q[1];
rz(0.3949365634637969) q[1];
ry(0.00011374890889292999) q[2];
rz(-2.9877173131009545) q[2];
ry(-0.00164029474940773) q[3];
rz(0.3704956943406899) q[3];
ry(3.1415774591464833) q[4];
rz(-0.40380289425353855) q[4];
ry(3.140819777886744) q[5];
rz(2.648560122768663) q[5];
ry(1.5686864668626288) q[6];
rz(1.8945856009846158) q[6];
ry(-0.000256046667193921) q[7];
rz(-0.7080925505980709) q[7];
ry(-3.1402641296410683) q[8];
rz(-0.9196080031989418) q[8];
ry(-1.57078657651197) q[9];
rz(1.332185095089268) q[9];
ry(-3.0903068645844827) q[10];
rz(3.028827196223953) q[10];
ry(-1.0184428526942935) q[11];
rz(0.004591437938575638) q[11];
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
ry(-1.5813053791979819) q[0];
rz(-1.4674406942198504) q[0];
ry(-3.125785356280189) q[1];
rz(-2.637795113608962) q[1];
ry(1.559890663789341) q[2];
rz(-1.4665524572930402) q[2];
ry(1.5845110851235913) q[3];
rz(-3.0328130030254803) q[3];
ry(3.091042370343054) q[4];
rz(-0.7733022202045654) q[4];
ry(-0.3412132880100849) q[5];
rz(-2.7900714572635046) q[5];
ry(-3.1077186719917833) q[6];
rz(2.049578347900371) q[6];
ry(1.5755601669574748) q[7];
rz(-3.0342626310197383) q[7];
ry(0.03529443568613022) q[8];
rz(3.0846991901486223) q[8];
ry(-0.019955764164626553) q[9];
rz(-1.2260854651409943) q[9];
ry(-1.580481465294649) q[10];
rz(-1.3813340353535712) q[10];
ry(1.5497837649666604) q[11];
rz(-1.4625349023547862) q[11];