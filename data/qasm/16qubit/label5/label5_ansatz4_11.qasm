OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-2.383385867510248) q[0];
rz(-0.5032480778768456) q[0];
ry(1.892678069987946) q[1];
rz(1.237502355393005) q[1];
ry(-1.1329232070841009) q[2];
rz(-3.1040887204596745) q[2];
ry(0.49961268015143256) q[3];
rz(-2.343947258223075) q[3];
ry(0.0010872194141995984) q[4];
rz(0.29312404632706784) q[4];
ry(-3.1408939993561438) q[5];
rz(-1.5175292601876555) q[5];
ry(3.1415756978799916) q[6];
rz(1.7132274834223338) q[6];
ry(-3.1415775781536524) q[7];
rz(-2.6781362165991673) q[7];
ry(0.9717262555138694) q[8];
rz(-1.5754798617067682) q[8];
ry(-1.1952818851780775) q[9];
rz(1.5784958151828423) q[9];
ry(-1.5699014838312104) q[10];
rz(-3.11455990367987) q[10];
ry(-3.141520209291066) q[11];
rz(-1.1268989901239221) q[11];
ry(0.19986672035120662) q[12];
rz(1.501092714108113) q[12];
ry(-1.5420348647390063) q[13];
rz(1.0632100208254094) q[13];
ry(1.6359864703848146) q[14];
rz(-0.003258789140908824) q[14];
ry(1.571669073565416) q[15];
rz(-3.1412161031631163) q[15];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-2.7974233501588452) q[0];
rz(-2.9072008549367343) q[0];
ry(0.31754313965126607) q[1];
rz(-2.2727111013344317) q[1];
ry(1.417424727236609) q[2];
rz(2.334662964704769) q[2];
ry(-1.617956723056284) q[3];
rz(-1.3475198228882699) q[3];
ry(-1.5713522317728341) q[4];
rz(-3.1392528184467543) q[4];
ry(1.57066393249523) q[5];
rz(0.0004818827135860468) q[5];
ry(-5.16699069067639e-05) q[6];
rz(0.2765801249037542) q[6];
ry(3.1415867363044083) q[7];
rz(-1.7690220981409466) q[7];
ry(0.0314691010197046) q[8];
rz(-1.568591281452174) q[8];
ry(0.15533627205944264) q[9];
rz(-1.578432309573797) q[9];
ry(3.141497577917719) q[10];
rz(0.3285302813421919) q[10];
ry(-1.5686919568990358) q[11];
rz(1.5863871860043446) q[11];
ry(-3.1089047444298763) q[12];
rz(-0.06962145391411044) q[12];
ry(-3.1402031694373846) q[13];
rz(-2.076454082564818) q[13];
ry(0.8519873721138076) q[14];
rz(2.5608098045163814) q[14];
ry(1.1843788505327846) q[15];
rz(1.5691376474469023) q[15];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.0762244376740566) q[0];
rz(1.3668598136474814) q[0];
ry(2.0154152965054286) q[1];
rz(1.1926485878849418) q[1];
ry(0.2879147828126145) q[2];
rz(-1.2781460769941537) q[2];
ry(-1.478287864564291) q[3];
rz(1.3731958285429438) q[3];
ry(-0.6707802480231928) q[4];
rz(-1.571514202490682) q[4];
ry(2.366407793809408) q[5];
rz(-0.9826810880416453) q[5];
ry(-3.1415688224530802) q[6];
rz(1.3542792566029003) q[6];
ry(3.1415563301380565) q[7];
rz(-0.9879982532076514) q[7];
ry(1.5705039280840545) q[8];
rz(3.1365476138198836) q[8];
ry(-1.5715666947673523) q[9];
rz(3.1387792456402304) q[9];
ry(-1.5711411570615823) q[10];
rz(1.3233230145450001) q[10];
ry(3.1398019883556705) q[11];
rz(-1.5576730241874517) q[11];
ry(1.5709590989924758) q[12];
rz(-1.5696086902566349) q[12];
ry(-3.141349974866013) q[13];
rz(1.5727227995257627) q[13];
ry(-0.07931763901653015) q[14];
rz(0.4022497995396117) q[14];
ry(1.556756456867598) q[15];
rz(1.2645478568779858) q[15];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(0.08763068765374223) q[0];
rz(1.57829882064361) q[0];
ry(-1.13405495712925) q[1];
rz(2.861517386412336) q[1];
ry(0.31965363965048676) q[2];
rz(1.5917841780192723) q[2];
ry(0.6564973496619456) q[3];
rz(-2.298851818697984) q[3];
ry(-1.674446767558448) q[4];
rz(0.001331335963310056) q[4];
ry(3.1404184938722826) q[5];
rz(0.5894196669726678) q[5];
ry(3.14153885803045) q[6];
rz(-3.1267606636320537) q[6];
ry(-3.1415921525419677) q[7];
rz(-2.1537869353463095) q[7];
ry(-2.246537170030268) q[8];
rz(1.5653123729118485) q[8];
ry(1.9008743801474905) q[9];
rz(3.140261556134575) q[9];
ry(0.00010684352616004676) q[10];
rz(-1.326618051002522) q[10];
ry(-1.569865510334452) q[11];
rz(0.8933152172880585) q[11];
ry(1.5418998454984227) q[12];
rz(1.9616426918913998) q[12];
ry(-1.5706016002998766) q[13];
rz(-0.5294001408136007) q[13];
ry(-3.102346313893494) q[14];
rz(1.4227314135929223) q[14];
ry(2.7815761610572594) q[15];
rz(-1.170390230754137) q[15];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(2.0453492604156667) q[0];
rz(0.9573429924583826) q[0];
ry(2.492961324524016) q[1];
rz(1.6242756659764868) q[1];
ry(1.1364616298391077) q[2];
rz(-3.007290424217129) q[2];
ry(-2.6993693387027786) q[3];
rz(1.4571802372367932) q[3];
ry(-1.5716789299821272) q[4];
rz(-1.1033220809054987) q[4];
ry(1.5689230095314066) q[5];
rz(2.646913640075656) q[5];
ry(-1.5713156089397593) q[6];
rz(0.6784097863504371) q[6];
ry(-1.5705250283816887) q[7];
rz(1.1438967586569009) q[7];
ry(-1.567273520553289) q[8];
rz(2.819787768913309) q[8];
ry(1.0185139528598692) q[9];
rz(-2.0141523138504223) q[9];
ry(-1.5445333665666299) q[10];
rz(-2.6797680761304594) q[10];
ry(-3.141552466005918) q[11];
rz(-2.248362929931143) q[11];
ry(3.140074068843689) q[12];
rz(1.9660178957839465) q[12];
ry(-0.0007270503040892607) q[13];
rz(1.167370673729632) q[13];
ry(-0.03519201713048136) q[14];
rz(1.8819589634957294) q[14];
ry(0.0468147585475398) q[15];
rz(-2.301733260844061) q[15];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-2.356203907855717) q[0];
rz(1.8238496013288734) q[0];
ry(-0.11671216127727406) q[1];
rz(2.830610325138718) q[1];
ry(-2.0732238068402387) q[2];
rz(-3.086914794744339) q[2];
ry(-0.8357022432633832) q[3];
rz(0.4070606768945426) q[3];
ry(-1.5705792025611671) q[4];
rz(-1.5707381159536715) q[4];
ry(-1.5706772725378773) q[5];
rz(1.5712154064676884) q[5];
ry(-1.2638600541745625e-05) q[6];
rz(-2.2488082311746243) q[6];
ry(-3.472389322236382e-05) q[7];
rz(-1.1471236338098496) q[7];
ry(-1.0404574721018297e-07) q[8];
rz(1.9842921040259058) q[8];
ry(-4.1079973524198915e-05) q[9];
rz(-2.559185727921892) q[9];
ry(3.141378225864304) q[10];
rz(-1.10893720174324) q[10];
ry(-1.5706409632108302) q[11];
rz(3.0934856752707787) q[11];
ry(1.5610840207660779) q[12];
rz(-3.140929367431299) q[12];
ry(-1.5860593886802983) q[13];
rz(0.6019556454749049) q[13];
ry(-0.02162516963176131) q[14];
rz(2.794680809426123) q[14];
ry(-1.2158908503312882) q[15];
rz(-1.8560595945593223) q[15];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.9511972631901406) q[0];
rz(-1.6976257949696798) q[0];
ry(0.7668011855007882) q[1];
rz(1.312753491004716) q[1];
ry(2.388682739721687) q[2];
rz(-1.691935751932765) q[2];
ry(0.5380751059935047) q[3];
rz(-1.2456822298143955) q[3];
ry(1.5699743075613224) q[4];
rz(-0.053436793798613984) q[4];
ry(1.5709898824588349) q[5];
rz(3.014942465013873) q[5];
ry(-0.625287923432151) q[6];
rz(0.6092633845911126) q[6];
ry(1.570675443198449) q[7];
rz(0.38952802791788166) q[7];
ry(-0.00013784330329968952) q[8];
rz(1.3348911699380874) q[8];
ry(3.1414080291092916) q[9];
rz(-0.4166053347944416) q[9];
ry(1.5711458990055256) q[10];
rz(-3.1405455951006167) q[10];
ry(3.141412356442928) q[11];
rz(2.9887012412106646) q[11];
ry(-0.3976685473267312) q[12];
rz(-1.1694341384632896) q[12];
ry(0.0007714808296084625) q[13];
rz(-0.6210261699716098) q[13];
ry(1.5731906263257356) q[14];
rz(-2.792082023453412) q[14];
ry(0.006584955734534574) q[15];
rz(1.925730373229798) q[15];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-1.9339400225668213) q[0];
rz(1.934636148069508) q[0];
ry(-0.9198384014414733) q[1];
rz(0.8188276205095957) q[1];
ry(-0.6133518162743963) q[2];
rz(2.928969226178035) q[2];
ry(2.226568582453774) q[3];
rz(0.664140851585576) q[3];
ry(-0.015333521454537369) q[4];
rz(1.4783331953757974) q[4];
ry(-2.8289703319975943) q[5];
rz(1.449845414345159) q[5];
ry(-0.0002692558814905155) q[6];
rz(2.531661786653733) q[6];
ry(-3.1415563366403845) q[7];
rz(-1.1669765535150258) q[7];
ry(3.141345003275971) q[8];
rz(2.751534851122276) q[8];
ry(0.0007776027863874909) q[9];
rz(-1.0167428663068168) q[9];
ry(1.570309135110051) q[10];
rz(-7.800805990143546e-05) q[10];
ry(0.0030913162986623277) q[11];
rz(-1.7408319532157017) q[11];
ry(3.1413682779571475) q[12];
rz(0.40059040618996805) q[12];
ry(-0.4321696878106618) q[13];
rz(0.00102160151843804) q[13];
ry(-0.0044162554645570395) q[14];
rz(-1.926014000622093) q[14];
ry(-3.051174619475122) q[15];
rz(-1.616256455792241) q[15];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(2.8182495597149644) q[0];
rz(-0.04130650791172031) q[0];
ry(2.735558495118603) q[1];
rz(-2.4844280429756207) q[1];
ry(0.0354134418030423) q[2];
rz(0.6338119105602438) q[2];
ry(-1.5794589877042762) q[3];
rz(0.15290579297797732) q[3];
ry(1.5708249535680938) q[4];
rz(-1.5710533750008762) q[4];
ry(1.5707083938036064) q[5];
rz(-0.05020712621076488) q[5];
ry(-1.5722148880412454) q[6];
rz(0.641016939363154) q[6];
ry(-1.5677467900590116) q[7];
rz(2.5718302901007695) q[7];
ry(1.573316784602917) q[8];
rz(1.3650210452144478) q[8];
ry(1.572267585040768) q[9];
rz(0.2563046587755658) q[9];
ry(1.5715554857619767) q[10];
rz(-1.5734660866037546) q[10];
ry(-3.141065001519835) q[11];
rz(0.4321666143601819) q[11];
ry(1.5706054633633513) q[12];
rz(1.5676571035918894) q[12];
ry(-2.1648348528689256) q[13];
rz(-0.0005279871417700476) q[13];
ry(-1.5701393527479643) q[14];
rz(1.1058117900025737) q[14];
ry(-1.5716773243832605) q[15];
rz(-1.5708904933265533) q[15];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.6441306157280362) q[0];
rz(-1.671662164317131) q[0];
ry(1.5760865229194878) q[1];
rz(1.581262514614693) q[1];
ry(-0.10596124876424229) q[2];
rz(-0.8824144881331635) q[2];
ry(1.4940919619583743) q[3];
rz(0.06395112633261135) q[3];
ry(1.5699646644674692) q[4];
rz(2.9839453609529483) q[4];
ry(0.05872126843686638) q[5];
rz(-3.0703154342842884) q[5];
ry(-3.14153214966324) q[6];
rz(-2.5770021994926826) q[6];
ry(-9.602459254409723e-07) q[7];
rz(2.140773733813103) q[7];
ry(3.1415669227321823) q[8];
rz(-0.1760991689380128) q[8];
ry(3.1413323376202427) q[9];
rz(-0.9016605644064104) q[9];
ry(0.2162471902262899) q[10];
rz(-0.2863906747254588) q[10];
ry(-3.1414807153677504) q[11];
rz(2.2847413119149187) q[11];
ry(-1.289834575243846) q[12];
rz(-0.19301184547640562) q[12];
ry(1.5697232012554938) q[13];
rz(2.1835246676409765) q[13];
ry(0.006395947325750823) q[14];
rz(-2.675875066785546) q[14];
ry(1.5686492369855074) q[15];
rz(-1.5727910215429493) q[15];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-2.7638385718386087) q[0];
rz(-2.1555459947289672) q[0];
ry(0.2030434597522184) q[1];
rz(1.1473300196238316) q[1];
ry(1.5707047271274268) q[2];
rz(-1.5704817294015734) q[2];
ry(1.5710703884874766) q[3];
rz(-1.5709713997335806) q[3];
ry(3.126710270495575) q[4];
rz(-0.10591846355090061) q[4];
ry(-3.101599841562858) q[5];
rz(0.02140983568220039) q[5];
ry(-5.822205772197187e-05) q[6];
rz(-0.7190076284282121) q[6];
ry(-1.5703628138075947) q[7];
rz(-3.122335268639636) q[7];
ry(0.6334419525096884) q[8];
rz(1.8016264205227503) q[8];
ry(-1.5567031935488995) q[9];
rz(0.29056914325187483) q[9];
ry(0.0003094106303072947) q[10];
rz(0.3629296336038842) q[10];
ry(-3.053556176003371) q[11];
rz(1.0621018631680874) q[11];
ry(3.141386385472796) q[12];
rz(0.7646119876115718) q[12];
ry(-0.0014592283553757435) q[13];
rz(2.529040297189041) q[13];
ry(2.7067237206214125) q[14];
rz(1.571696916376716) q[14];
ry(0.3934853408848876) q[15];
rz(-1.793471018595365) q[15];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(-3.1405940097363674) q[0];
rz(1.071827414576792) q[0];
ry(-3.1337365835573854) q[1];
rz(2.72522713309668) q[1];
ry(1.7811707811151756) q[2];
rz(-1.573565266197056) q[2];
ry(1.552956788445158) q[3];
rz(2.9619572581533857) q[3];
ry(1.5618845904830205) q[4];
rz(-1.9270801698565374) q[4];
ry(1.5703746328754555) q[5];
rz(-1.7484193517741877) q[5];
ry(3.131536205844399) q[6];
rz(-0.4531668924745916) q[6];
ry(-3.14010747824849) q[7];
rz(-1.528874914824689) q[7];
ry(-0.005297557585250983) q[8];
rz(0.3430963451441604) q[8];
ry(-3.1414119920162213) q[9];
rz(1.8335280034777395) q[9];
ry(-0.00015498367104438415) q[10];
rz(-3.0249239843442544) q[10];
ry(-0.0003016944204210503) q[11];
rz(-2.554663590865322) q[11];
ry(1.4185093006223317) q[12];
rz(1.352166221274529) q[12];
ry(-1.5703045166133767) q[13];
rz(3.0066490890283335) q[13];
ry(-1.571012287258452) q[14];
rz(-1.9518324246085674) q[14];
ry(-0.002435154809669271) q[15];
rz(0.21483110758032842) q[15];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(2.1951888266007686) q[0];
rz(0.022583047943473475) q[0];
ry(-1.5722803806222059) q[1];
rz(0.6438577829319369) q[1];
ry(-2.7775829009078508) q[2];
rz(-1.6095687935560745) q[2];
ry(0.0004734733923719147) q[3];
rz(1.752197492668622) q[3];
ry(-3.141584692255395) q[4];
rz(-1.9318834441034725) q[4];
ry(2.687027391489594) q[5];
rz(3.1287272921933926) q[5];
ry(3.139423762195133) q[6];
rz(-2.274448860989761) q[6];
ry(-2.4160093034935417) q[7];
rz(1.9843877927263862) q[7];
ry(3.1220845472008123) q[8];
rz(0.39794859289043827) q[8];
ry(-0.8009965670184228) q[9];
rz(-0.9317578161006402) q[9];
ry(-0.0035586994278826722) q[10];
rz(3.01331353193665) q[10];
ry(-0.00012086501412451624) q[11];
rz(0.841341883422086) q[11];
ry(-1.6484229575303633) q[12];
rz(-0.005481094695598763) q[12];
ry(3.1408296379317733) q[13];
rz(-0.13367406912567945) q[13];
ry(-3.1410591538329338) q[14];
rz(2.913915827652877) q[14];
ry(-0.0008872845671658293) q[15];
rz(-1.560813768062826) q[15];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.5681652420967647) q[0];
rz(0.07184246642407643) q[0];
ry(1.5697462497195822) q[1];
rz(0.003843027368576556) q[1];
ry(-3.004743721531207) q[2];
rz(2.8574192647420573) q[2];
ry(1.5707063318992782) q[3];
rz(1.5752373888357218) q[3];
ry(-1.5767993337072643) q[4];
rz(-1.5691356475849816) q[4];
ry(-1.570469140942131) q[5];
rz(0.3828833947871271) q[5];
ry(0.020066026717793232) q[6];
rz(-2.020197086061858) q[6];
ry(-3.1392792099951334) q[7];
rz(1.9974389850742897) q[7];
ry(-3.135139448749458) q[8];
rz(-2.4384176182047455) q[8];
ry(-0.0004139055501822334) q[9];
rz(2.5309829337730307) q[9];
ry(1.6619871230361127e-05) q[10];
rz(3.0794006699691625) q[10];
ry(-3.140589564066584) q[11];
rz(-1.0549142067452273) q[11];
ry(-2.5114499939642263) q[12];
rz(-0.4087302822743472) q[12];
ry(-1.5704948370628777) q[13];
rz(-0.19490712692226533) q[13];
ry(-3.1401925744193577) q[14];
rz(1.3282920261274764) q[14];
ry(1.5708659509629546) q[15];
rz(-1.570105895963426) q[15];
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
cz q[0],q[3];
cz q[1],q[2];
cz q[2],q[5];
cz q[3],q[4];
cz q[4],q[7];
cz q[5],q[6];
cz q[6],q[9];
cz q[7],q[8];
cz q[8],q[11];
cz q[9],q[10];
cz q[10],q[13];
cz q[11],q[12];
cz q[12],q[15];
cz q[13],q[14];
ry(1.5701862095729178) q[0];
rz(-2.874376230163296) q[0];
ry(1.5714634443329771) q[1];
rz(-1.7788865775499894) q[1];
ry(-0.0009765241249404255) q[2];
rz(2.086455106009687) q[2];
ry(-1.5697807675982223) q[3];
rz(2.9354230866311117) q[3];
ry(-1.5708904634559513) q[4];
rz(-2.8740303855948164) q[4];
ry(3.1409723971925962) q[5];
rz(-1.389236238516049) q[5];
ry(3.0469193151225427) q[6];
rz(-2.7986108342801126) q[6];
ry(1.5710247164867237) q[7];
rz(2.942818925208869) q[7];
ry(-0.06312317699605607) q[8];
rz(-2.2179806088963456) q[8];
ry(1.5796054793732341) q[9];
rz(-2.3171635319559827) q[9];
ry(-1.5706168845092368) q[10];
rz(1.8243757599079116) q[10];
ry(3.1415473881116407) q[11];
rz(2.3826112495199316) q[11];
ry(1.5700877507332667) q[12];
rz(1.8242905123171205) q[12];
ry(-3.1411653667808803) q[13];
rz(-2.1300609040703633) q[13];
ry(3.141406858777011) q[14];
rz(-0.14800304513237839) q[14];
ry(-1.571321022723195) q[15];
rz(2.7760179750413445) q[15];