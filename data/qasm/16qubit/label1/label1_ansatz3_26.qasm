OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(3.097899779479624) q[0];
rz(-0.17257839817306664) q[0];
ry(1.6164276174814711) q[1];
rz(1.4284571647893394) q[1];
ry(-1.6104038249133983) q[2];
rz(-1.632366578237117) q[2];
ry(-1.4269739885850536) q[3];
rz(-2.7007085395417354) q[3];
ry(1.7065551845464608) q[4];
rz(-1.8241230773086665) q[4];
ry(0.16002324387157163) q[5];
rz(-3.103922845940022) q[5];
ry(-1.54541767011353) q[6];
rz(-1.5320267025702055) q[6];
ry(3.1401930930389947) q[7];
rz(1.091927741876874) q[7];
ry(-0.0009686211520802956) q[8];
rz(0.5623470194494926) q[8];
ry(0.7970388339835184) q[9];
rz(-1.440066600839254) q[9];
ry(-0.8506949825629988) q[10];
rz(2.8823148238016363) q[10];
ry(3.1409708212310004) q[11];
rz(-1.7564590615653302) q[11];
ry(-2.7419081730624497) q[12];
rz(-1.1709810017395657) q[12];
ry(1.8678549903732455) q[13];
rz(-0.01330342388573058) q[13];
ry(-2.0141146105339143) q[14];
rz(0.6350201060736205) q[14];
ry(2.4462597574664677) q[15];
rz(1.3532577435544266) q[15];
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
ry(1.6918915612810457) q[0];
rz(1.3034987605996575) q[0];
ry(0.2941334431603142) q[1];
rz(-1.8856887533060531) q[1];
ry(1.318348895695193) q[2];
rz(-1.2281403521710734) q[2];
ry(-0.0019606200446744496) q[3];
rz(2.2599669238014624) q[3];
ry(0.016511359419611793) q[4];
rz(-1.0564683134567767) q[4];
ry(3.1392355980774407) q[5];
rz(2.5319212006519196) q[5];
ry(-0.2572284234180105) q[6];
rz(0.8506289817540665) q[6];
ry(1.645044049998665) q[7];
rz(2.2229551742269713) q[7];
ry(1.3298759895815107) q[8];
rz(-3.077659731773115) q[8];
ry(-2.448158424230243) q[9];
rz(1.711274332436231) q[9];
ry(0.2675270521353603) q[10];
rz(-1.4903127013150925) q[10];
ry(-0.0005709918091647975) q[11];
rz(-1.053509465994123) q[11];
ry(-2.8058152800005027) q[12];
rz(0.58770910952558) q[12];
ry(1.8410887905231412) q[13];
rz(-2.650873070386601) q[13];
ry(2.976770541177499) q[14];
rz(-1.1331069466620352) q[14];
ry(-1.4646021569455454) q[15];
rz(-1.813733841102015) q[15];
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
ry(-1.6472993096715074) q[0];
rz(2.8253655158500233) q[0];
ry(1.1986910097224635) q[1];
rz(1.5750762917066785) q[1];
ry(-1.4861033173112927) q[2];
rz(-0.5155626097689971) q[2];
ry(-1.082374635869007) q[3];
rz(-2.0779842473408237) q[3];
ry(1.1885598545421412) q[4];
rz(2.4318022842389575) q[4];
ry(-0.001662110951845541) q[5];
rz(-2.7034639701866623) q[5];
ry(3.12933273961922) q[6];
rz(2.3062342190542298) q[6];
ry(-3.1326418327499854) q[7];
rz(-2.142666056075395) q[7];
ry(-3.138974318793087) q[8];
rz(2.105157110740996) q[8];
ry(0.8600359179341048) q[9];
rz(1.8598233580480161) q[9];
ry(1.7183542460840662) q[10];
rz(-0.5040278700176692) q[10];
ry(-0.0004601711367031028) q[11];
rz(2.1234786146443243) q[11];
ry(0.35209764899716883) q[12];
rz(1.049737981291135) q[12];
ry(3.098231145568174) q[13];
rz(2.880351207115697) q[13];
ry(-0.7846862302522136) q[14];
rz(1.4291438868241004) q[14];
ry(1.3036906879642989) q[15];
rz(0.554020471447562) q[15];
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
ry(2.9156183330418797) q[0];
rz(2.7839740785211378) q[0];
ry(0.05313146912679756) q[1];
rz(-2.586646755515788) q[1];
ry(0.18024811847256486) q[2];
rz(0.33393102464303953) q[2];
ry(3.059160669888304) q[3];
rz(2.7895171839184933) q[3];
ry(0.10444858794291179) q[4];
rz(-0.2656655951191977) q[4];
ry(3.1036436095685476) q[5];
rz(0.5669568890807629) q[5];
ry(-1.7305476195596026) q[6];
rz(-2.415260343839255) q[6];
ry(1.5912069186131585) q[7];
rz(0.05215321476827927) q[7];
ry(0.09707934394439394) q[8];
rz(-0.9758326325862158) q[8];
ry(-2.926689728043419) q[9];
rz(-1.2422182878276633) q[9];
ry(2.2144624742865133) q[10];
rz(-0.9611670488930549) q[10];
ry(-0.000735131504652209) q[11];
rz(-2.5990430384817107) q[11];
ry(-0.40945118089126536) q[12];
rz(1.56417534262504) q[12];
ry(2.4243965720302514) q[13];
rz(-1.2801964103379193) q[13];
ry(-1.602480928888685) q[14];
rz(2.552549224297048) q[14];
ry(2.5344647948170786) q[15];
rz(2.7495780584167417) q[15];
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
ry(1.8251786312803806) q[0];
rz(1.1870521103315932) q[0];
ry(-1.210615280018702) q[1];
rz(2.8638508946661085) q[1];
ry(-1.7543404855584965) q[2];
rz(-2.5920802384317634) q[2];
ry(-2.7724736906473275) q[3];
rz(-2.6056804993555986) q[3];
ry(-0.16672176108209058) q[4];
rz(1.4079228109445463) q[4];
ry(3.141487256700718) q[5];
rz(-0.7181589949216185) q[5];
ry(0.0006848566887428346) q[6];
rz(1.6541538854133693) q[6];
ry(0.0705126913140793) q[7];
rz(1.4499289639067987) q[7];
ry(3.1267757489378853) q[8];
rz(1.8989128225652587) q[8];
ry(-2.1536938452362397) q[9];
rz(-2.5031645359818873) q[9];
ry(-2.112571750174499) q[10];
rz(1.8025312985982973) q[10];
ry(-0.00098263837224413) q[11];
rz(1.914123324151509) q[11];
ry(-2.94675204175634) q[12];
rz(-2.1892004272957166) q[12];
ry(-1.832334644171845) q[13];
rz(-3.0920798327510193) q[13];
ry(0.8536236229406463) q[14];
rz(-1.5219050885023155) q[14];
ry(-2.6661363939459735) q[15];
rz(1.4239614372415468) q[15];
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
ry(-0.023797704668907294) q[0];
rz(0.7657469579617047) q[0];
ry(1.5764925523253757) q[1];
rz(-0.024103214086421413) q[1];
ry(-0.6333624573094928) q[2];
rz(1.6889201279766581) q[2];
ry(0.015292353749557286) q[3];
rz(-2.65244394496747) q[3];
ry(2.611944382301763) q[4];
rz(-0.4927376901415324) q[4];
ry(-0.0012855009460759348) q[5];
rz(-1.3012046334741871) q[5];
ry(-0.21021235516106795) q[6];
rz(-2.683617317759352) q[6];
ry(-1.8607298837440511) q[7];
rz(-1.3758162923245716) q[7];
ry(0.0022909336180152455) q[8];
rz(2.646885530718588) q[8];
ry(-1.278406618206529) q[9];
rz(-0.16467896145622127) q[9];
ry(1.6810077607483918) q[10];
rz(-1.4499577250877058) q[10];
ry(-0.0003261479567897824) q[11];
rz(2.5110241276739798) q[11];
ry(0.00993242089750891) q[12];
rz(0.14457489387144487) q[12];
ry(-1.563806650866696) q[13];
rz(1.4515048079675812) q[13];
ry(2.2604787243566786) q[14];
rz(-2.314160408744754) q[14];
ry(2.8456904850187192) q[15];
rz(-1.9227519045420822) q[15];
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
ry(-0.014412989226445749) q[0];
rz(-1.8772268630428268) q[0];
ry(0.548265166169795) q[1];
rz(-1.53973479365854) q[1];
ry(0.04325173500808965) q[2];
rz(-2.973441101026985) q[2];
ry(0.02563193544229936) q[3];
rz(1.5344991715137524) q[3];
ry(-0.037538715110708935) q[4];
rz(-1.2453569391226562) q[4];
ry(-2.9876031664428) q[5];
rz(-2.166306146343813) q[5];
ry(-3.137263162349781) q[6];
rz(-0.5444650953352522) q[6];
ry(2.527717219725435) q[7];
rz(1.3896021912803027) q[7];
ry(2.8902883488531996) q[8];
rz(-1.2200022012396072) q[8];
ry(2.843585929370571) q[9];
rz(1.1534751683300133) q[9];
ry(-2.459894039417655) q[10];
rz(1.608070429554215) q[10];
ry(-3.1404847365298796) q[11];
rz(-0.41404228171294233) q[11];
ry(-1.5090821110348043) q[12];
rz(1.4890459068557442) q[12];
ry(-0.5541734581043709) q[13];
rz(-1.9043478287044708) q[13];
ry(-1.8343386025642077) q[14];
rz(1.2619400786143928) q[14];
ry(-2.018726792728123) q[15];
rz(-2.325994630111161) q[15];
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
ry(1.8782160463638509) q[0];
rz(-1.6292619475861034) q[0];
ry(-2.187816834328737) q[1];
rz(-3.0361559272858565) q[1];
ry(2.3617170105614895) q[2];
rz(2.13655477812022) q[2];
ry(-3.1353661678465565) q[3];
rz(-2.996026607813807) q[3];
ry(0.5108995450818421) q[4];
rz(1.6132016008731165) q[4];
ry(0.03204364006296955) q[5];
rz(-1.8301014859175015) q[5];
ry(3.1287289175453803) q[6];
rz(-2.909285010974933) q[6];
ry(-0.0226623455712982) q[7];
rz(-1.5934427041959864) q[7];
ry(1.1868667626259093) q[8];
rz(0.30678664246173604) q[8];
ry(1.8961279001839237) q[9];
rz(2.9516715278705816) q[9];
ry(-3.115693782175676) q[10];
rz(2.3251786055982624) q[10];
ry(-0.013530139475631309) q[11];
rz(0.2952738728502311) q[11];
ry(0.0018749265271560496) q[12];
rz(-1.286873396942344) q[12];
ry(-1.079001455830995) q[13];
rz(3.0752678333081978) q[13];
ry(1.4348719510528334) q[14];
rz(-1.8244511725769577) q[14];
ry(-0.6519921488211137) q[15];
rz(0.09779704849543691) q[15];
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
ry(0.09028183114375389) q[0];
rz(-2.271159148400647) q[0];
ry(-1.3906759636560109) q[1];
rz(-1.623422940396396) q[1];
ry(3.1366132858063143) q[2];
rz(-1.7072201946020975) q[2];
ry(0.0370918394099391) q[3];
rz(-2.5703629916323876) q[3];
ry(-0.014645225865092816) q[4];
rz(1.937689337097131) q[4];
ry(-0.06860739936664562) q[5];
rz(2.92921625279458) q[5];
ry(-0.0005160748449615227) q[6];
rz(1.6883380525507892) q[6];
ry(1.614415201869449) q[7];
rz(-1.4215972805246357) q[7];
ry(-3.1382548354171678) q[8];
rz(1.9548405285925015) q[8];
ry(-0.05449613673808197) q[9];
rz(-1.4019423244813738) q[9];
ry(3.1048648811323156) q[10];
rz(2.105733136782143) q[10];
ry(0.004917621901407543) q[11];
rz(2.318430082057282) q[11];
ry(0.37136415926868316) q[12];
rz(0.2841498429543385) q[12];
ry(-2.6847733608396767) q[13];
rz(0.665726865620007) q[13];
ry(2.578017723890192) q[14];
rz(1.1464398037420747) q[14];
ry(2.23399244903582) q[15];
rz(1.5348055420210098) q[15];
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
ry(-2.0439820561383106) q[0];
rz(-2.806934274923077) q[0];
ry(-3.1116255201150533) q[1];
rz(0.8744329039030161) q[1];
ry(3.123665713573312) q[2];
rz(-0.9323755155861639) q[2];
ry(1.6039705777487754) q[3];
rz(2.320331084194948) q[3];
ry(0.11544243777357507) q[4];
rz(-2.001779435232911) q[4];
ry(3.1046446458617902) q[5];
rz(-3.0822644709395344) q[5];
ry(0.011637006395605812) q[6];
rz(2.4357742810579848) q[6];
ry(0.045920069618417436) q[7];
rz(0.8096043462281748) q[7];
ry(1.6502991141432877) q[8];
rz(2.576248982569949) q[8];
ry(2.045075143695258) q[9];
rz(-0.9770263178282953) q[9];
ry(0.037921995442106934) q[10];
rz(0.4914540877460851) q[10];
ry(-0.006032862375109048) q[11];
rz(-1.9834541331482314) q[11];
ry(-3.0965455473090393) q[12];
rz(2.782853855788473) q[12];
ry(2.714213260079827) q[13];
rz(1.27977069020059) q[13];
ry(0.5780334313177891) q[14];
rz(-1.0874519514240744) q[14];
ry(-0.8279167225684061) q[15];
rz(-0.7328214406282809) q[15];
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
ry(-0.0926520846966614) q[0];
rz(-3.080310120855874) q[0];
ry(0.0017601632574386628) q[1];
rz(3.059400792802427) q[1];
ry(0.013266645164760466) q[2];
rz(-0.7875990145654095) q[2];
ry(0.09643673571479283) q[3];
rz(2.3376817163724066) q[3];
ry(0.5159130621336807) q[4];
rz(1.4530162954964163) q[4];
ry(1.5972197159643124) q[5];
rz(-1.0136871613894132) q[5];
ry(3.141213953475732) q[6];
rz(1.7348162109721574) q[6];
ry(-0.8065149015799185) q[7];
rz(-1.06089729196486) q[7];
ry(-0.7784010170581865) q[8];
rz(0.008613383313049283) q[8];
ry(-1.5871123549214463) q[9];
rz(-1.06533698174686) q[9];
ry(-3.086622414939601) q[10];
rz(-2.4320060425072554) q[10];
ry(0.05949420503985614) q[11];
rz(-0.1526655244124493) q[11];
ry(-2.162753199260282) q[12];
rz(1.9733906686919287) q[12];
ry(-1.289017655615658) q[13];
rz(2.1054820227596096) q[13];
ry(-0.44619963187244205) q[14];
rz(0.42420907550676373) q[14];
ry(-1.7713803249606261) q[15];
rz(0.1455778329700601) q[15];
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
ry(1.6748565065926106) q[0];
rz(2.2764492685469393) q[0];
ry(-3.0652708374382476) q[1];
rz(0.8694245006183524) q[1];
ry(0.625703037621228) q[2];
rz(-0.7397491725526218) q[2];
ry(1.567486330882856) q[3];
rz(3.081446491645268) q[3];
ry(-3.140521295355268) q[4];
rz(-2.6371632238474314) q[4];
ry(-0.004499890557874583) q[5];
rz(2.613019678088244) q[5];
ry(0.00015929537257530768) q[6];
rz(3.0104856755594223) q[6];
ry(0.00014584223745927203) q[7];
rz(1.6691280178885166) q[7];
ry(1.5831271750177103) q[8];
rz(3.0879855392195728) q[8];
ry(-2.9645905365100638) q[9];
rz(-2.7365259470166805) q[9];
ry(0.010339775640448768) q[10];
rz(-0.09453542520337582) q[10];
ry(-3.141484769693994) q[11];
rz(-1.3669253659244767) q[11];
ry(-0.17970684540276316) q[12];
rz(-2.342685736601422) q[12];
ry(-0.43933113791226663) q[13];
rz(0.5051543825976585) q[13];
ry(-0.22627857360701853) q[14];
rz(-2.05801541736398) q[14];
ry(0.3561326486238973) q[15];
rz(2.451689068313188) q[15];
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
ry(-2.353328109689387) q[0];
rz(2.800390174564408) q[0];
ry(-0.013645533369417429) q[1];
rz(1.4590560996761388) q[1];
ry(-3.090518681788765) q[2];
rz(-0.5184027025372986) q[2];
ry(-1.5856426777127177) q[3];
rz(1.303074162117192) q[3];
ry(-1.5091985244830803) q[4];
rz(2.0289154216768175) q[4];
ry(1.5632614551458328) q[5];
rz(1.0028573929219788) q[5];
ry(-3.107232331456919) q[6];
rz(2.833604235991531) q[6];
ry(0.3216675184519566) q[7];
rz(1.2025401696287046) q[7];
ry(0.7856715778671878) q[8];
rz(-1.6025609002016945) q[8];
ry(-1.4320988142551725) q[9];
rz(-2.8946577305083747) q[9];
ry(-2.7145457150049195) q[10];
rz(0.08198074404266897) q[10];
ry(-3.126923711197291) q[11];
rz(-1.1278704542299312) q[11];
ry(-0.791615821328971) q[12];
rz(0.8272110728766623) q[12];
ry(-1.8688398155104382) q[13];
rz(-1.197132194692292) q[13];
ry(-0.9417843387910212) q[14];
rz(2.8439999158705773) q[14];
ry(-2.0391766857728197) q[15];
rz(1.8457899771862314) q[15];
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
ry(-0.22860477341631746) q[0];
rz(2.3285154360662825) q[0];
ry(2.025897135798333) q[1];
rz(3.128029721487021) q[1];
ry(0.11719741872075085) q[2];
rz(2.8057659542257194) q[2];
ry(-1.3101615877935773) q[3];
rz(-2.6190682576404347) q[3];
ry(-0.09283706251988384) q[4];
rz(-1.5315368122041386) q[4];
ry(-3.0978349692296967) q[5];
rz(-0.3811756513505911) q[5];
ry(-3.124883612443541) q[6];
rz(1.8844317213044253) q[6];
ry(0.20578913893997136) q[7];
rz(0.00454435083639293) q[7];
ry(-2.3914456717730834) q[8];
rz(2.30708256277677) q[8];
ry(0.09015814205395187) q[9];
rz(2.994473459154496) q[9];
ry(-0.09913976566913223) q[10];
rz(0.2113218446118195) q[10];
ry(-0.01807154829673685) q[11];
rz(-0.18001840476735423) q[11];
ry(3.133454750370873) q[12];
rz(0.7743950240265942) q[12];
ry(-0.9759719137809142) q[13];
rz(2.6476514784651) q[13];
ry(2.131762562211754) q[14];
rz(-2.001197016540492) q[14];
ry(-1.0830989021865174) q[15];
rz(-1.0877002558901199) q[15];
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
ry(-0.02518044234660787) q[0];
rz(-0.7577313679552313) q[0];
ry(0.07972580586972426) q[1];
rz(3.135235904134802) q[1];
ry(-2.91335008267685) q[2];
rz(1.8432798715650185) q[2];
ry(3.135920889239798) q[3];
rz(2.7405552993565654) q[3];
ry(0.37232201012253774) q[4];
rz(-2.046776403853201) q[4];
ry(-0.09707308667965542) q[5];
rz(-1.060792119319891) q[5];
ry(-0.031214056480183494) q[6];
rz(-1.79974789003861) q[6];
ry(-1.5459567394671287) q[7];
rz(-1.1552746581370243) q[7];
ry(-0.03832293226480465) q[8];
rz(0.3149778141004171) q[8];
ry(0.11270438981287446) q[9];
rz(-3.074580650212992) q[9];
ry(-0.7918479813922491) q[10];
rz(2.6481071830492353) q[10];
ry(0.034902558152464166) q[11];
rz(2.9278010995027253) q[11];
ry(-2.602241094237221) q[12];
rz(2.3320175867275443) q[12];
ry(-1.6382799646952035) q[13];
rz(0.8802023243572847) q[13];
ry(0.4809419490751725) q[14];
rz(-0.7081221381647744) q[14];
ry(-1.3700062960321393) q[15];
rz(1.678329787981331) q[15];
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
ry(0.9826400665795131) q[0];
rz(-1.4630190604124362) q[0];
ry(1.8800258743515021) q[1];
rz(1.888133693339726) q[1];
ry(0.11819966886400128) q[2];
rz(-0.8663760400880948) q[2];
ry(0.391103494658043) q[3];
rz(3.1243200856258455) q[3];
ry(-2.7383399941487694) q[4];
rz(2.744742927777641) q[4];
ry(1.612852173248432) q[5];
rz(-2.6551432786968747) q[5];
ry(2.0526257951191673) q[6];
rz(-2.7619413678720472) q[6];
ry(0.38613978135739574) q[7];
rz(2.781310836733264) q[7];
ry(3.00992768843876) q[8];
rz(3.0270697418582984) q[8];
ry(3.0385733138680893) q[9];
rz(0.10320108582331623) q[9];
ry(-3.081114027320111) q[10];
rz(-0.46678598074074923) q[10];
ry(-3.1381205704917394) q[11];
rz(-1.8780064792736644) q[11];
ry(-3.061850259724452) q[12];
rz(3.0802764115151122) q[12];
ry(1.145426815839164) q[13];
rz(1.4568388249040964) q[13];
ry(0.5472365052589887) q[14];
rz(2.8600460495810958) q[14];
ry(-1.8185132461745956) q[15];
rz(3.141348837752125) q[15];
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
ry(1.415592841157845) q[0];
rz(-1.430660947372056) q[0];
ry(1.9908729923421085) q[1];
rz(0.03892923896308442) q[1];
ry(-0.5181451844284088) q[2];
rz(-0.003138957227627006) q[2];
ry(0.014720053778042006) q[3];
rz(0.28399584527506244) q[3];
ry(-1.3404243928020865) q[4];
rz(1.7722797962019703) q[4];
ry(-0.03652348591726093) q[5];
rz(-1.5070954889252342) q[5];
ry(0.07300383143619182) q[6];
rz(-1.8842742419932916) q[6];
ry(-3.129337794247099) q[7];
rz(-0.20730573957767273) q[7];
ry(-0.00036798445173502614) q[8];
rz(2.2454075005643848) q[8];
ry(-3.0903849764341786) q[9];
rz(1.6651766276155966) q[9];
ry(-2.2561433580712897) q[10];
rz(1.4221427089075922) q[10];
ry(-3.1091785148212954) q[11];
rz(-0.48937486437167077) q[11];
ry(-1.4117489054583814) q[12];
rz(-2.996521143555507) q[12];
ry(1.4861465483721494) q[13];
rz(0.5480337627428955) q[13];
ry(0.18496699648146406) q[14];
rz(-0.18896181054098452) q[14];
ry(1.185868885768235) q[15];
rz(-2.5467694010789996) q[15];
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
ry(-2.3830228567453804) q[0];
rz(1.8773451391751474) q[0];
ry(-0.11135739585304784) q[1];
rz(-0.7501005255545651) q[1];
ry(-3.111538971898841) q[2];
rz(-0.14208964719089234) q[2];
ry(-3.106250347005378) q[3];
rz(-1.9456511483355081) q[3];
ry(0.020221406157299518) q[4];
rz(1.3980333155754863) q[4];
ry(1.5719823892699054) q[5];
rz(-1.2873465502877677) q[5];
ry(1.549415549934733) q[6];
rz(0.10812245343670383) q[6];
ry(-1.971877234105321) q[7];
rz(2.097051592686812) q[7];
ry(-2.1004580830124855) q[8];
rz(1.860205310665605) q[8];
ry(-2.935430448706676) q[9];
rz(0.7664223892232513) q[9];
ry(2.233710760053456) q[10];
rz(-0.31699247754033283) q[10];
ry(1.5404835339506917) q[11];
rz(1.4009566898500703) q[11];
ry(1.4824692140675106) q[12];
rz(-0.3310640394237821) q[12];
ry(1.2854570731211776) q[13];
rz(1.4458565060308468) q[13];
ry(2.0133048792908994) q[14];
rz(-3.06738566527682) q[14];
ry(-3.1236671351998564) q[15];
rz(-2.483020070091397) q[15];
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
ry(-0.4563287462048855) q[0];
rz(1.3034307298685346) q[0];
ry(0.42213069133798875) q[1];
rz(-2.6844724173613788) q[1];
ry(-2.539860351861534) q[2];
rz(2.7419817287086325) q[2];
ry(-2.5178730911771163) q[3];
rz(-1.6259503875279004) q[3];
ry(1.471827705174662) q[4];
rz(0.25692520093994897) q[4];
ry(-2.9407765439606655) q[5];
rz(0.4186670687474391) q[5];
ry(-3.1233665218831725) q[6];
rz(-1.5097102069185468) q[6];
ry(-0.0182490125215761) q[7];
rz(-1.2858952582911027) q[7];
ry(-0.005083394540714714) q[8];
rz(-2.4071459083718576) q[8];
ry(-0.0021638420540694976) q[9];
rz(-0.1540595090332369) q[9];
ry(-3.1298007765292835) q[10];
rz(-3.0251384545783173) q[10];
ry(1.3365782961603596) q[11];
rz(-1.6262917551236837) q[11];
ry(-3.1268363588315813) q[12];
rz(-0.642685218605078) q[12];
ry(2.809251033532344) q[13];
rz(1.4978545458802683) q[13];
ry(-0.04990525012529352) q[14];
rz(-2.709659720244946) q[14];
ry(-2.3900637848842163) q[15];
rz(0.8714167817735997) q[15];
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
ry(2.2721024513584096) q[0];
rz(-1.8890432685768803) q[0];
ry(-1.4253120310294742) q[1];
rz(-0.22371475158164425) q[1];
ry(3.052654750260621) q[2];
rz(2.5836958940789416) q[2];
ry(3.137949111309333) q[3];
rz(-1.7435375574766585) q[3];
ry(0.19787286206527332) q[4];
rz(1.46002556495102) q[4];
ry(-1.5794709989917675) q[5];
rz(3.045805829024092) q[5];
ry(1.5853272133841112) q[6];
rz(1.9887508645240977) q[6];
ry(-0.5988770723349008) q[7];
rz(-2.3162768839362102) q[7];
ry(2.538097868658065) q[8];
rz(1.3138409253891066) q[8];
ry(-0.0848294310250591) q[9];
rz(1.3972222811763786) q[9];
ry(2.8413611817706337) q[10];
rz(2.5326768133177167) q[10];
ry(1.4666683942065872) q[11];
rz(-2.2233091830944955) q[11];
ry(0.24111925336328177) q[12];
rz(3.12715170756693) q[12];
ry(-2.9364773218029785) q[13];
rz(1.1271414816513863) q[13];
ry(1.6329838964279642) q[14];
rz(2.647007830892273) q[14];
ry(-0.01102653419850963) q[15];
rz(-2.315801260106442) q[15];
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
ry(-1.3436291823139586) q[0];
rz(2.205376747295607) q[0];
ry(2.4801965140139037) q[1];
rz(1.1725383228633441) q[1];
ry(-0.40193871539016657) q[2];
rz(-0.11051632124353508) q[2];
ry(3.135599839891594) q[3];
rz(0.490977346425838) q[3];
ry(-1.6095723723268032) q[4];
rz(1.5878033915231695) q[4];
ry(2.947786705973753) q[5];
rz(1.4868803648871118) q[5];
ry(0.004844497769307751) q[6];
rz(0.07350182731550206) q[6];
ry(-0.005664762851729476) q[7];
rz(2.4059908705495827) q[7];
ry(3.1262797309876427) q[8];
rz(0.2614526843488157) q[8];
ry(3.137679376745007) q[9];
rz(-0.2237764454936189) q[9];
ry(0.05972311894038285) q[10];
rz(3.0969088925596635) q[10];
ry(-2.832272086626409) q[11];
rz(-2.7765854223553905) q[11];
ry(3.139817618967439) q[12];
rz(1.7381970686762385) q[12];
ry(1.3684196524557954) q[13];
rz(0.2275474132781518) q[13];
ry(-1.7691426675171504) q[14];
rz(-1.654400756000657) q[14];
ry(-1.023974550676841) q[15];
rz(1.4775861352607527) q[15];
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
ry(-0.36712412611402234) q[0];
rz(2.3155453555858627) q[0];
ry(1.8166702251330857) q[1];
rz(-1.0918066190743647) q[1];
ry(-0.006820923765147313) q[2];
rz(-0.7928757322999543) q[2];
ry(-0.06503340563748163) q[3];
rz(2.3550247446991026) q[3];
ry(1.417337849998678) q[4];
rz(0.04275115594053691) q[4];
ry(1.6186308579730166) q[5];
rz(-2.5094021002847984) q[5];
ry(3.0450972628212547) q[6];
rz(-1.4510686234149945) q[6];
ry(1.8784749096953615) q[7];
rz(-0.8059342414898181) q[7];
ry(-0.7986797499928241) q[8];
rz(-2.985058458956101) q[8];
ry(-1.1460984486659989) q[9];
rz(-0.9139469181398131) q[9];
ry(1.905295183709626) q[10];
rz(2.0088147725882433) q[10];
ry(2.677697878765087) q[11];
rz(-0.4740221880826574) q[11];
ry(3.075106541395629) q[12];
rz(0.14348022340515953) q[12];
ry(-1.620038405021516) q[13];
rz(-1.5290198403285338) q[13];
ry(-1.526153819178039) q[14];
rz(-1.7112176786820372) q[14];
ry(3.1406637976864933) q[15];
rz(0.11494386808599355) q[15];
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
ry(1.4054025985630982) q[0];
rz(-0.10594648316926736) q[0];
ry(0.15187430701869312) q[1];
rz(-1.3775641701754449) q[1];
ry(0.21236057240394055) q[2];
rz(-0.12549766230039733) q[2];
ry(3.127787428751079) q[3];
rz(-0.018197742366193325) q[3];
ry(-1.6577636263200004) q[4];
rz(1.8278475496553899) q[4];
ry(-3.059177119672282) q[5];
rz(1.840996718871106) q[5];
ry(-3.1408282920313297) q[6];
rz(2.0943209317706053) q[6];
ry(3.1411616893719736) q[7];
rz(-2.4853755214354387) q[7];
ry(-3.1121971922118723) q[8];
rz(-1.06821364452981) q[8];
ry(0.026708253651538175) q[9];
rz(-2.0781514593681) q[9];
ry(3.0441037394525345) q[10];
rz(-1.1593719143594612) q[10];
ry(0.0007439267018547469) q[11];
rz(-1.8662252664317185) q[11];
ry(2.9978575073372102) q[12];
rz(2.8362957185770914) q[12];
ry(1.6371032419763019) q[13];
rz(1.7352976320122384) q[13];
ry(0.4647188947721904) q[14];
rz(-0.29165653132389036) q[14];
ry(-3.137702199684357) q[15];
rz(-2.1064918539098443) q[15];
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
ry(-3.129966612239459) q[0];
rz(2.6390514549948008) q[0];
ry(-2.1138322683926507) q[1];
rz(2.8507858898292606) q[1];
ry(1.606701012922355) q[2];
rz(1.6155825013448464) q[2];
ry(-1.1905060447961802) q[3];
rz(-0.33020130130006725) q[3];
ry(0.19391780768621658) q[4];
rz(-1.648155664696965) q[4];
ry(-1.7301759964400274) q[5];
rz(3.092490936161633) q[5];
ry(2.5472862217535406) q[6];
rz(2.643847772918764) q[6];
ry(1.6336452750074661) q[7];
rz(-0.39441720979643774) q[7];
ry(-2.0744209480234517) q[8];
rz(2.1350372705521017) q[8];
ry(-1.2222966162285447) q[9];
rz(-1.1147183340004911) q[9];
ry(2.0087383891114143) q[10];
rz(-2.4133691992773025) q[10];
ry(3.119219291692778) q[11];
rz(-1.4457542974669924) q[11];
ry(-0.8116503415738623) q[12];
rz(0.48979853194936596) q[12];
ry(2.35870096072087) q[13];
rz(-3.121755301682711) q[13];
ry(-0.23783873273799913) q[14];
rz(-0.7317965132804449) q[14];
ry(-0.515609894978159) q[15];
rz(-0.48280511826700123) q[15];
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
ry(1.5464811632829392) q[0];
rz(0.16378051298695162) q[0];
ry(3.1331631428939937) q[1];
rz(3.071493114156973) q[1];
ry(-2.4842308738927454) q[2];
rz(-1.4677188077699341) q[2];
ry(-0.0054728650889392955) q[3];
rz(-2.5776448904279436) q[3];
ry(-0.02509049732559413) q[4];
rz(2.9538278071827095) q[4];
ry(0.02042230066731321) q[5];
rz(-2.936396156314237) q[5];
ry(3.1336337949556077) q[6];
rz(-1.055300230279105) q[6];
ry(0.013976597129941376) q[7];
rz(-2.4477240580807686) q[7];
ry(-3.129916142366664) q[8];
rz(1.8870545416826747) q[8];
ry(-3.1405698509473066) q[9];
rz(2.243488264272287) q[9];
ry(-3.1167521357161503) q[10];
rz(1.67137829729883) q[10];
ry(-1.5674209177148624) q[11];
rz(-1.5673864990013362) q[11];
ry(3.1093818897514605) q[12];
rz(2.3321927807559866) q[12];
ry(-0.04058704375995752) q[13];
rz(2.859103776926329) q[13];
ry(-0.8168403335677116) q[14];
rz(1.5226566359634557) q[14];
ry(3.132472471188093) q[15];
rz(-0.9313992344884131) q[15];
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
ry(1.250273613428704) q[0];
rz(-1.4935337690802566) q[0];
ry(1.504386871767653) q[1];
rz(-1.5637420821685462) q[1];
ry(0.632473524099078) q[2];
rz(-1.12089910062432) q[2];
ry(-2.5697380894913704) q[3];
rz(2.4918556643879604) q[3];
ry(3.0747481663533667) q[4];
rz(0.028720115177520246) q[4];
ry(-0.30095591291156326) q[5];
rz(-2.018179557080184) q[5];
ry(-2.3428954013490335) q[6];
rz(0.9472377384823868) q[6];
ry(2.8247711777147013) q[7];
rz(-2.7415477023915695) q[7];
ry(2.321247329620197) q[8];
rz(-1.05466927946176) q[8];
ry(3.140097961463226) q[9];
rz(1.9441047396639695) q[9];
ry(-0.003436207040236283) q[10];
rz(1.022128452578921) q[10];
ry(-1.3467216377551567) q[11];
rz(0.020666096774328024) q[11];
ry(2.9699649254701264) q[12];
rz(-2.44727517356552) q[12];
ry(-1.566555815566467) q[13];
rz(2.293964807623442) q[13];
ry(1.3940420395772817) q[14];
rz(-0.8598272007563637) q[14];
ry(-1.1434764980201162) q[15];
rz(1.771170387761253) q[15];
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
ry(1.3392811262943458) q[0];
rz(-1.453183022347935) q[0];
ry(2.816033072310442) q[1];
rz(3.0653822573956284) q[1];
ry(3.066196874041349) q[2];
rz(0.1904012818011564) q[2];
ry(0.002178850081846484) q[3];
rz(-0.7475536281193499) q[3];
ry(0.04976605022823788) q[4];
rz(1.93734072802666) q[4];
ry(-3.140312241030557) q[5];
rz(0.854169373647803) q[5];
ry(-0.0027966874004761166) q[6];
rz(-0.9526157121789763) q[6];
ry(3.120462649991613) q[7];
rz(-2.574611477296312) q[7];
ry(0.014227734218942346) q[8];
rz(2.481471262399979) q[8];
ry(0.015471171973467812) q[9];
rz(-0.6188643771551963) q[9];
ry(0.009210083807977118) q[10];
rz(0.947396903508757) q[10];
ry(1.5963322241153657) q[11];
rz(-0.7267648067056276) q[11];
ry(-0.006663186931688441) q[12];
rz(1.3698058115301175) q[12];
ry(-3.122721226154148) q[13];
rz(1.6799128601556712) q[13];
ry(-0.13570118537982562) q[14];
rz(-2.6386283147642415) q[14];
ry(-2.100053008433211) q[15];
rz(0.5534512085800734) q[15];
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
ry(0.0059203853317515) q[0];
rz(0.735675636340334) q[0];
ry(1.8666339578493956) q[1];
rz(1.7405667117226382) q[1];
ry(-3.0913591054618887) q[2];
rz(1.2965046362115338) q[2];
ry(1.3931468093044284) q[3];
rz(-2.08176944883938) q[3];
ry(1.8201141007154165) q[4];
rz(1.5783815330622382) q[4];
ry(1.9885239433475697) q[5];
rz(1.6832337186672268) q[5];
ry(-2.89203450781588) q[6];
rz(0.9282462803180493) q[6];
ry(-2.9500324864079066) q[7];
rz(-0.3231644824500997) q[7];
ry(-0.9417452160541414) q[8];
rz(-0.7559337677231683) q[8];
ry(-2.184216956069273) q[9];
rz(2.7052767590702578) q[9];
ry(1.152531100901191) q[10];
rz(0.06589214914713414) q[10];
ry(-0.8288889301980253) q[11];
rz(2.0777835258044446) q[11];
ry(-0.6154357669827892) q[12];
rz(-2.8693924652517557) q[12];
ry(1.5364390477698429) q[13];
rz(-1.641480721102143) q[13];
ry(-3.1096905861895614) q[14];
rz(-0.613259170600891) q[14];
ry(0.16927764687847263) q[15];
rz(-2.130206901526896) q[15];
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
ry(-2.274994476298314) q[0];
rz(-1.1660522145487642) q[0];
ry(-2.8754208530022587) q[1];
rz(2.272410544964372) q[1];
ry(-0.03376308697702157) q[2];
rz(-3.0850904165553366) q[2];
ry(0.00506021399220824) q[3];
rz(-2.3630646397592963) q[3];
ry(0.05798375556155579) q[4];
rz(2.6141558054567957) q[4];
ry(-0.028741375586407592) q[5];
rz(0.8175721680403534) q[5];
ry(-0.001380966866540696) q[6];
rz(-1.3195047303344245) q[6];
ry(-3.1350183492466774) q[7];
rz(-1.5962855446891402) q[7];
ry(0.0659419062080697) q[8];
rz(-2.716584740270804) q[8];
ry(3.1318313899673837) q[9];
rz(-1.9031573200962475) q[9];
ry(-3.12902765023609) q[10];
rz(-2.8331547070470164) q[10];
ry(0.0010504976648403174) q[11];
rz(-0.7770794184451644) q[11];
ry(3.136674326022488) q[12];
rz(-1.1803416356417804) q[12];
ry(3.0677411579249116) q[13];
rz(-3.0330872905549198) q[13];
ry(1.4782024409436185) q[14];
rz(-2.588573680517657) q[14];
ry(1.6274195400493516) q[15];
rz(-0.07509051536391922) q[15];
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
ry(0.01308955040821103) q[0];
rz(-1.1864004408864277) q[0];
ry(-2.9898033481114834) q[1];
rz(1.7918388458709593) q[1];
ry(-1.6027420017249554) q[2];
rz(2.887397545190422) q[2];
ry(-1.642950829541395) q[3];
rz(-0.2071873543259146) q[3];
ry(-2.5875895696855835) q[4];
rz(0.971902164302823) q[4];
ry(1.3608569996634516) q[5];
rz(-0.3745466110313452) q[5];
ry(1.1675164203033441) q[6];
rz(-1.9475456103882287) q[6];
ry(-0.43505826085951416) q[7];
rz(2.2193221011868474) q[7];
ry(-1.0834982634098262) q[8];
rz(-1.8451548103678772) q[8];
ry(3.090753209732506) q[9];
rz(-0.0955439133199344) q[9];
ry(-0.22242685851615998) q[10];
rz(1.2501680550027072) q[10];
ry(-1.1755437410692835) q[11];
rz(1.380511140702367) q[11];
ry(-0.1292277345491037) q[12];
rz(1.4202120510286302) q[12];
ry(0.08956966997628779) q[13];
rz(-1.9802498117537015) q[13];
ry(-3.0852345304295867) q[14];
rz(1.7972719758078857) q[14];
ry(-0.2750479258276366) q[15];
rz(-1.7265070023039304) q[15];