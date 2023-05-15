OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(2.769721483835063) q[0];
rz(-2.748421341260646) q[0];
ry(-2.549831515297772) q[1];
rz(-2.639264398799423) q[1];
ry(0.02300185903178163) q[2];
rz(-0.5340020676084887) q[2];
ry(-0.0008277401020810871) q[3];
rz(2.160208272249834) q[3];
ry(0.09952778422341389) q[4];
rz(1.1559974705606937) q[4];
ry(3.004351810157885) q[5];
rz(-1.4889675987975348) q[5];
ry(-3.1075053796769176) q[6];
rz(0.22263439435018473) q[6];
ry(3.139802282097724) q[7];
rz(2.6716764359198217) q[7];
ry(1.57112917121623) q[8];
rz(2.3943168938011095) q[8];
ry(-1.570942967975392) q[9];
rz(3.14112205223749) q[9];
ry(3.141578056473846) q[10];
rz(-1.884731024643775) q[10];
ry(0.63329790819831) q[11];
rz(1.5702167958776876) q[11];
ry(-1.5704942253870742) q[12];
rz(1.4661949393834788) q[12];
ry(-1.5706688517704208) q[13];
rz(0.01335868575960276) q[13];
ry(3.141565677075508) q[14];
rz(-1.6005907472196368) q[14];
ry(-0.00016150183845287103) q[15];
rz(1.2646912970741981) q[15];
ry(1.48212252656347) q[16];
rz(-3.1264976556645707) q[16];
ry(-1.3069464727840545) q[17];
rz(0.5385405451464712) q[17];
ry(-3.1406164781026433) q[18];
rz(-1.088541140252264) q[18];
ry(3.1373948872154322) q[19];
rz(2.104825015562712) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.7683162364869993) q[0];
rz(-2.6582183463257474) q[0];
ry(-2.0325264467462705) q[1];
rz(2.321736297234204) q[1];
ry(0.036696650440891425) q[2];
rz(-2.130443138498559) q[2];
ry(-0.04707473730224669) q[3];
rz(-2.9280981493483167) q[3];
ry(-1.5616126188208668) q[4];
rz(1.3830055036521611) q[4];
ry(-1.5951094383440865) q[5];
rz(-1.326882976760344) q[5];
ry(-0.36645491940378694) q[6];
rz(3.0668507917167798) q[6];
ry(-1.5713822282675063) q[7];
rz(2.791982677109161) q[7];
ry(-3.140429944784006) q[8];
rz(-0.35275852323136236) q[8];
ry(1.1134609411798013) q[9];
rz(1.6640012311508334) q[9];
ry(3.1407882000449394) q[10];
rz(-2.216372725803306) q[10];
ry(1.5731150803023415) q[11];
rz(-2.278022163005013) q[11];
ry(0.0028654295698593273) q[12];
rz(1.5551332949692012) q[12];
ry(1.5594746634640648) q[13];
rz(2.3680935352145576) q[13];
ry(0.0004052147979761541) q[14];
rz(1.2680979341551593) q[14];
ry(3.1412949865493385) q[15];
rz(2.8280710966007385) q[15];
ry(1.5444656501283154) q[16];
rz(2.1746010486950187) q[16];
ry(0.06935854821729735) q[17];
rz(2.4715212382745153) q[17];
ry(0.9991298164312282) q[18];
rz(-2.7725328897312154) q[18];
ry(-1.0876357771527623) q[19];
rz(-2.8011244457590694) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-2.107731250574272) q[0];
rz(1.0570150864502672) q[0];
ry(-2.191904086401987) q[1];
rz(-0.6759580360340198) q[1];
ry(1.5532956354185012) q[2];
rz(-0.1766438221294144) q[2];
ry(1.644743234406703) q[3];
rz(1.1787961427804214) q[3];
ry(0.013698086498021114) q[4];
rz(1.7605773339210098) q[4];
ry(3.0527231553336915) q[5];
rz(-2.295937366752152) q[5];
ry(1.3575894226550558) q[6];
rz(1.6705552790186757) q[6];
ry(-3.068425700148485) q[7];
rz(-2.4260678348596256) q[7];
ry(-0.3441142252888518) q[8];
rz(-0.39555063473906915) q[8];
ry(3.0282714437019407) q[9];
rz(0.2379925818562567) q[9];
ry(1.783851919098458) q[10];
rz(1.0747803058295196) q[10];
ry(-2.053724452704155) q[11];
rz(0.040895663659952675) q[11];
ry(-0.9356250143912259) q[12];
rz(0.04442117789191702) q[12];
ry(2.227992326497113) q[13];
rz(-1.9958961622757505) q[13];
ry(-1.5664453126249014) q[14];
rz(1.570827744813445) q[14];
ry(0.0066276243887668064) q[15];
rz(-3.0429532526444842) q[15];
ry(3.1028351397927776) q[16];
rz(-1.6662019682116287) q[16];
ry(0.27255592675113416) q[17];
rz(0.08812957354462644) q[17];
ry(0.533046895129603) q[18];
rz(-0.4115076054413464) q[18];
ry(-0.6586535187394511) q[19];
rz(2.614100887533348) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(2.9168301853480236) q[0];
rz(1.9544354652174563) q[0];
ry(0.023868760358404018) q[1];
rz(2.506363366957432) q[1];
ry(0.05494512963312698) q[2];
rz(-1.3009717588279903) q[2];
ry(0.09278082657291886) q[3];
rz(-2.7839518440755384) q[3];
ry(-0.005250452107073922) q[4];
rz(0.5738128895784556) q[4];
ry(0.00048092833080026764) q[5];
rz(2.077105077404109) q[5];
ry(0.15463359275918004) q[6];
rz(-2.2187180228068435) q[6];
ry(3.130100677982169) q[7];
rz(1.0662687467883545) q[7];
ry(3.139624909899741) q[8];
rz(2.616895949253582) q[8];
ry(0.0087127704460368) q[9];
rz(-0.18829214723902418) q[9];
ry(3.1384118780028345) q[10];
rz(0.09756305111551083) q[10];
ry(0.0016917780362428426) q[11];
rz(1.670663851155639) q[11];
ry(-3.1415898799864057) q[12];
rz(-1.7089197327271703) q[12];
ry(0.0001356282182676637) q[13];
rz(1.0973751026023582) q[13];
ry(-1.5302385234885438) q[14];
rz(2.0866817395894435) q[14];
ry(1.5740496374942534) q[15];
rz(3.1403528028059635) q[15];
ry(-2.9813213980044617) q[16];
rz(-0.7421314305394678) q[16];
ry(1.6902386263634683) q[17];
rz(0.6063631957501379) q[17];
ry(2.7708296688852734) q[18];
rz(1.8190407259416135) q[18];
ry(-0.4042760206352898) q[19];
rz(-1.4843451361934532) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.050036084913178805) q[0];
rz(1.9776541062815056) q[0];
ry(-3.123520723335524) q[1];
rz(0.28467494360239004) q[1];
ry(-3.0688903384994566) q[2];
rz(-1.1907638493607866) q[2];
ry(-0.7526352517371928) q[3];
rz(1.6737341714058926) q[3];
ry(-0.029441554042487775) q[4];
rz(-1.0402938677564686) q[4];
ry(-3.10892053437433) q[5];
rz(-1.5409181239731646) q[5];
ry(0.27778636643343146) q[6];
rz(-2.535401422443335) q[6];
ry(1.493001449053311) q[7];
rz(0.029042718234159954) q[7];
ry(2.769557748102303) q[8];
rz(2.5637097287172033) q[8];
ry(2.9755966970036125) q[9];
rz(0.0045884490485806765) q[9];
ry(-0.3671501009653299) q[10];
rz(-2.1948843040046038) q[10];
ry(-1.8360417337351957) q[11];
rz(2.5165940254378314) q[11];
ry(-1.5708335420974167) q[12];
rz(-2.0752722286056366) q[12];
ry(-0.00038959899345147875) q[13];
rz(-0.6053276356372956) q[13];
ry(1.569764053976885) q[14];
rz(-0.6555978816443463) q[14];
ry(1.5717785846126642) q[15];
rz(-2.404334934505287) q[15];
ry(1.5686845768610445) q[16];
rz(3.1362994236075883) q[16];
ry(-3.1398524902870437) q[17];
rz(-0.8573308137118287) q[17];
ry(-3.141244306220832) q[18];
rz(1.9323088386484777) q[18];
ry(-3.140412448810619) q[19];
rz(-1.1780543285013498) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(1.8811826773326006) q[0];
rz(-0.4613607131974575) q[0];
ry(-2.8553482558777508) q[1];
rz(0.2455350463130188) q[1];
ry(-1.717437381022191) q[2];
rz(-0.9807120440893113) q[2];
ry(-1.6025558280833312) q[3];
rz(-1.404100803917502) q[3];
ry(-3.141168700760008) q[4];
rz(1.9173569390619793) q[4];
ry(0.0010140441884614049) q[5];
rz(-0.7071293442242202) q[5];
ry(1.5612621022258935) q[6];
rz(-0.04056246946666775) q[6];
ry(1.59419383809009) q[7];
rz(-0.003408034614892586) q[7];
ry(-3.0304918816455317) q[8];
rz(3.080013516652713) q[8];
ry(1.1639647769957016) q[9];
rz(-1.6914949543473288) q[9];
ry(1.5712569670607133) q[10];
rz(-1.4858244700277297) q[10];
ry(-3.141106443058775) q[11];
rz(2.412623652011087) q[11];
ry(0.00031034001627316797) q[12];
rz(0.5045153443803266) q[12];
ry(0.00027805347310147597) q[13];
rz(0.49086891012364736) q[13];
ry(0.00038818643383155315) q[14];
rz(2.6503788790252316) q[14];
ry(-3.14117168998869) q[15];
rz(-1.8118679033736615) q[15];
ry(-0.9230773471640904) q[16];
rz(2.9905060504598717) q[16];
ry(-1.5461857051142391) q[17];
rz(0.0007822053890125247) q[17];
ry(-1.5724549360881515) q[18];
rz(1.5377288694211333) q[18];
ry(-0.002648864417544416) q[19];
rz(0.8586422099575284) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(3.075172791672872) q[0];
rz(-2.4965914864705363) q[0];
ry(3.0287083826263124) q[1];
rz(-2.6603677238284837) q[1];
ry(3.066967697571504) q[2];
rz(-1.0392211343324833) q[2];
ry(-0.06847335414880881) q[3];
rz(-1.6798307380188264) q[3];
ry(0.011889974800665117) q[4];
rz(-0.7543032283378424) q[4];
ry(3.1263008219793993) q[5];
rz(-1.3462204814251064) q[5];
ry(-3.0822249999515816) q[6];
rz(1.555939335275312) q[6];
ry(-1.6130047200916238) q[7];
rz(0.0010126229245987829) q[7];
ry(1.5710399400022572) q[8];
rz(1.6260708409545241) q[8];
ry(-1.5708579332212196) q[9];
rz(1.6232180924063817) q[9];
ry(-3.1369980315535906) q[10];
rz(0.08501096369058737) q[10];
ry(-0.00037455338028538203) q[11];
rz(2.481713634455256) q[11];
ry(1.5706001797948375) q[12];
rz(2.9536802576264427) q[12];
ry(-1.570125012179233) q[13];
rz(2.311375307401469) q[13];
ry(3.1397224014121488) q[14];
rz(1.991825482198487) q[14];
ry(1.5461228645518046) q[15];
rz(-0.007009231460951842) q[15];
ry(-0.0027200131849226474) q[16];
rz(-2.385419146033773) q[16];
ry(1.5646505976994431) q[17];
rz(3.1355388140755553) q[17];
ry(-2.954342421927864) q[18];
rz(1.9319511061337538) q[18];
ry(0.046481053929868565) q[19];
rz(0.7343147313081781) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.9936095058201768) q[0];
rz(0.4822072738674734) q[0];
ry(3.129698221221879) q[1];
rz(2.462420806728927) q[1];
ry(-1.567439812636748) q[2];
rz(1.810579220113751) q[2];
ry(1.6203911464471878) q[3];
rz(1.6929533806035986) q[3];
ry(-0.003115685845726061) q[4];
rz(0.032271826536435455) q[4];
ry(-3.139595267096328) q[5];
rz(-2.5778028515870957) q[5];
ry(-1.5554660270791223) q[6];
rz(-3.1388515286854997) q[6];
ry(-1.566291323122142) q[7];
rz(-0.0007922942601783908) q[7];
ry(0.04363446245577312) q[8];
rz(3.085633836593638) q[8];
ry(-3.1181410271978343) q[9];
rz(-3.0888118314550947) q[9];
ry(-1.5694564907362099) q[10];
rz(-2.7505731528703334) q[10];
ry(-8.9161290043549e-05) q[11];
rz(-1.3052239594952952) q[11];
ry(0.0010451844963570117) q[12];
rz(1.7136848915135303) q[12];
ry(-0.0009957917098027723) q[13];
rz(-1.9872456793129725) q[13];
ry(0.4260773400052361) q[14];
rz(1.5750822562957565) q[14];
ry(3.1296807955006467) q[15];
rz(-1.5489797538828638) q[15];
ry(0.00021174828005854351) q[16];
rz(2.869696271233047) q[16];
ry(-1.5739779548207125) q[17];
rz(1.3564958998410714) q[17];
ry(-3.141557843933694) q[18];
rz(-1.1091195465564492) q[18];
ry(0.0007331891760991027) q[19];
rz(2.473019673947285) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(2.0347773499891013) q[0];
rz(0.9883022063225958) q[0];
ry(2.432597403022014) q[1];
rz(-0.4001538716921686) q[1];
ry(-1.597081037392686) q[2];
rz(-1.5975993087738536) q[2];
ry(-0.03275263092957766) q[3];
rz(1.6031905489666176) q[3];
ry(-1.5688757991781719) q[4];
rz(2.1078113495567123) q[4];
ry(1.566486356580118) q[5];
rz(-2.5714329355123446) q[5];
ry(-1.6152294767071873) q[6];
rz(2.0670085470083146) q[6];
ry(-1.5770556819640003) q[7];
rz(-0.04808249297060396) q[7];
ry(-1.5729078010299329) q[8];
rz(0.10872596986299662) q[8];
ry(1.6060361930814935) q[9];
rz(-2.9814206089623196) q[9];
ry(-0.0035687191802357522) q[10];
rz(1.3218388211075602) q[10];
ry(-0.7380827725620565) q[11];
rz(-0.0003903840649243608) q[11];
ry(-0.0010219473732995255) q[12];
rz(1.6151683914198678) q[12];
ry(3.1410766325022856) q[13];
rz(0.32517804925854826) q[13];
ry(1.5717145373676393) q[14];
rz(-1.6653270531116533) q[14];
ry(1.5697036538482507) q[15];
rz(1.5764352132211066) q[15];
ry(0.002136673063498049) q[16];
rz(0.958570761787481) q[16];
ry(3.0874891402613422) q[17];
rz(-3.03012338099249) q[17];
ry(-1.5688285724486315) q[18];
rz(3.1381524377736723) q[18];
ry(1.3885022314843494) q[19];
rz(-1.5707781516347543) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(0.24462941219303844) q[0];
rz(1.7849746414916867) q[0];
ry(-2.887767970712013) q[1];
rz(2.1330985203343977) q[1];
ry(1.2404818895736731) q[2];
rz(-3.10939636284678) q[2];
ry(0.36192378275071135) q[3];
rz(1.4667319575898219) q[3];
ry(-1.5868781240736352) q[4];
rz(-2.0821685711775118) q[4];
ry(1.5796234089461212) q[5];
rz(-1.0940289775863663) q[5];
ry(0.05878729472874588) q[6];
rz(-2.0314030784053267) q[6];
ry(-0.160217732559083) q[7];
rz(-2.728314207823297) q[7];
ry(3.13110656518641) q[8];
rz(0.11063412995674948) q[8];
ry(0.006386575861252908) q[9];
rz(1.4104940691215706) q[9];
ry(-3.141258080682445) q[10];
rz(-1.4290342762850408) q[10];
ry(-1.5691207862574932) q[11];
rz(-3.1407493534375592) q[11];
ry(1.5703797951978211) q[12];
rz(-0.061364493588390176) q[12];
ry(1.5709158413140765) q[13];
rz(-1.570945572214475) q[13];
ry(0.010260542913944076) q[14];
rz(0.09258415713043001) q[14];
ry(2.7160523713770726) q[15];
rz(3.124150428301635) q[15];
ry(0.0029745151704782054) q[16];
rz(0.27992184649615665) q[16];
ry(-0.0018805035059640002) q[17];
rz(1.2463701023804983) q[17];
ry(1.5684544116153047) q[18];
rz(1.605795438036428) q[18];
ry(1.5709698308241729) q[19];
rz(-2.5760469996069335) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(2.8112170342594966) q[0];
rz(-0.2699023585853205) q[0];
ry(2.697398167536604) q[1];
rz(-3.023169614953196) q[1];
ry(-3.139358525051982) q[2];
rz(-1.4267810987274565) q[2];
ry(1.5654267590701298) q[3];
rz(1.61340547590937) q[3];
ry(0.00026338280420734606) q[4];
rz(1.9511737715023854) q[4];
ry(-0.002887327689374448) q[5];
rz(3.0463819462630215) q[5];
ry(1.5807458365298634) q[6];
rz(-2.33392810226617) q[6];
ry(-3.1413549513705603) q[7];
rz(0.24993942039531974) q[7];
ry(-1.5696857459486546) q[8];
rz(-1.8567191526978108) q[8];
ry(1.570828272815088) q[9];
rz(2.5078766246348008) q[9];
ry(-1.9529174115187795) q[10];
rz(0.20378551873682985) q[10];
ry(1.5720316410622326) q[11];
rz(-0.00013732618123483568) q[11];
ry(3.141591571114745) q[12];
rz(-3.061979590909694) q[12];
ry(2.474439869889385) q[13];
rz(-0.3373605694683226) q[13];
ry(-1.5533752858247682) q[14];
rz(1.2153613839713042) q[14];
ry(-1.5698352547914352) q[15];
rz(0.47732290285818524) q[15];
ry(1.571136433935882) q[16];
rz(-3.06435936310386) q[16];
ry(-0.5692568657915484) q[17];
rz(-0.0032944077346827513) q[17];
ry(2.9609054589107444) q[18];
rz(-3.069070588322992) q[18];
ry(1.533755993355316) q[19];
rz(1.6194231574197644) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.5794273998525528) q[0];
rz(3.1396803141988783) q[0];
ry(1.5619167245207528) q[1];
rz(-1.6131540865374845) q[1];
ry(1.6150052027524557) q[2];
rz(1.5749372259992158) q[2];
ry(1.568903947499943) q[3];
rz(1.5080986878972786) q[3];
ry(1.5674160701030713) q[4];
rz(1.1995401603508276) q[4];
ry(-0.008947681282250208) q[5];
rz(2.237611981276034) q[5];
ry(0.0012487364123921314) q[6];
rz(0.7530565520728087) q[6];
ry(-3.140854766653267) q[7];
rz(1.6810920934817364) q[7];
ry(0.000774552195825784) q[8];
rz(1.8568953075876748) q[8];
ry(3.14147392496433) q[9];
rz(-0.6824855026841758) q[9];
ry(-0.00023864444899679427) q[10];
rz(-0.20362996567730737) q[10];
ry(-1.5692999087129884) q[11];
rz(2.6586030176103144) q[11];
ry(-3.141379492227675) q[12];
rz(-1.2427748969554244) q[12];
ry(3.1414661927268672) q[13];
rz(2.8034121975191177) q[13];
ry(-0.00018594785565095862) q[14];
rz(0.3565050852771945) q[14];
ry(0.00018447599801518777) q[15];
rz(2.663448519075657) q[15];
ry(-2.6254483506368502) q[16];
rz(-1.0418367990248631) q[16];
ry(-1.5704733078555235) q[17];
rz(-1.987167134441852) q[17];
ry(-3.1274152264679778) q[18];
rz(0.1461061666464846) q[18];
ry(-1.572662130074607) q[19];
rz(1.8384955052585086) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-1.5702135867758171) q[0];
rz(0.04748751102089699) q[0];
ry(0.04054531487736215) q[1];
rz(1.616611449472473) q[1];
ry(-1.5738026628878126) q[2];
rz(-0.0001642933665660253) q[2];
ry(-1.5708960486307353) q[3];
rz(-0.008293265469387913) q[3];
ry(-3.1414713891707158) q[4];
rz(1.2275646287660178) q[4];
ry(0.00014253821364018648) q[5];
rz(-1.9697085796903988) q[5];
ry(-1.7129301418406637) q[6];
rz(-2.6048237816090842) q[6];
ry(0.06775740908989705) q[7];
rz(1.2642003183670738) q[7];
ry(1.5703578030668874) q[8];
rz(-1.9789475557735627) q[8];
ry(0.0803897702092051) q[9];
rz(1.6936900268505983) q[9];
ry(-1.5683214349763794) q[10];
rz(-3.139657752724353) q[10];
ry(3.140473361537249) q[11];
rz(-2.3515860940753464) q[11];
ry(3.052905445588475e-05) q[12];
rz(-1.512516331901366) q[12];
ry(1.5727868947640318) q[13];
rz(-3.141471348415795) q[13];
ry(-1.5805652259640188) q[14];
rz(-2.1810595829671944) q[14];
ry(-1.5697281335884123) q[15];
rz(3.1336876063824524) q[15];
ry(-3.1415336586928375) q[16];
rz(-0.8428504851747786) q[16];
ry(0.0011644537921480236) q[17];
rz(1.977386632714512) q[17];
ry(-3.1410378642970302) q[18];
rz(-1.4628771012101005) q[18];
ry(1.5712173107238623) q[19];
rz(0.3259802134477401) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-3.1386703357577006) q[0];
rz(1.5530748793420965) q[0];
ry(1.2311857885586786) q[1];
rz(-1.5738621839914655) q[1];
ry(-1.5705131587817318) q[2];
rz(-1.5484361129108086) q[2];
ry(-1.5698174156579876) q[3];
rz(-2.5135582191329244) q[3];
ry(-2.8116603392206327) q[4];
rz(3.13041337135853) q[4];
ry(-3.1378497752708916) q[5];
rz(0.6688514294109318) q[5];
ry(3.141371573853365) q[6];
rz(0.5404477839381059) q[6];
ry(3.141472247159059) q[7];
rz(1.5539484893137354) q[7];
ry(3.1413808096286164) q[8];
rz(-1.9819347553756108) q[8];
ry(-3.1414105208267276) q[9];
rz(-3.0897596116407335) q[9];
ry(-1.570769488034901) q[10];
rz(-1.5712089732250414) q[10];
ry(0.007735178862408878) q[11];
rz(2.4346457809195443) q[11];
ry(3.13566003750883) q[12];
rz(1.2379509371127941) q[12];
ry(-1.5715475892249415) q[13];
rz(-1.5710933396445679) q[13];
ry(6.307963624974599e-05) q[14];
rz(-2.498887000887609) q[14];
ry(-0.04581195183366415) q[15];
rz(1.5829203887788095) q[15];
ry(-1.5701641967632023) q[16];
rz(-2.8951877373858284) q[16];
ry(-0.048947176892309756) q[17];
rz(0.009573209894700188) q[17];
ry(1.5710480763715655) q[18];
rz(1.5702881256605856) q[18];
ry(3.131815069066498) q[19];
rz(-0.9018032079561743) q[19];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[16],q[17];
cz q[18],q[19];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[6],q[8];
cz q[8],q[10];
cz q[10],q[12];
cz q[12],q[14];
cz q[14],q[16];
cz q[16],q[18];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
cz q[7],q[9];
cz q[9],q[11];
cz q[11],q[13];
cz q[13],q[15];
cz q[15],q[17];
cz q[17],q[19];
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
cz q[14],q[17];
cz q[15],q[16];
cz q[16],q[19];
cz q[17],q[18];
ry(-2.1635155269369286) q[0];
rz(-2.953867743381192) q[0];
ry(1.5714524806974242) q[1];
rz(0.2911915257867941) q[1];
ry(1.576220029977129) q[2];
rz(1.8069493244396266) q[2];
ry(0.001314220796788652) q[3];
rz(1.1302048583555155) q[3];
ry(-1.6002584122132806) q[4];
rz(-2.910209664897078) q[4];
ry(-1.570046084264927) q[5];
rz(0.18143825485152124) q[5];
ry(1.683617822175992) q[6];
rz(-2.916133074164025) q[6];
ry(1.5752867900278096) q[7];
rz(-1.37401300827781) q[7];
ry(-0.7058439160009408) q[8];
rz(0.2353804487717213) q[8];
ry(3.141226369764933) q[9];
rz(0.15652851247733765) q[9];
ry(-1.5709637960407938) q[10];
rz(0.23145448726876466) q[10];
ry(-3.1410032056798975) q[11];
rz(-0.8241547873135335) q[11];
ry(-3.1414301327156355) q[12];
rz(1.2242307222516897) q[12];
ry(1.5708018037141978) q[13];
rz(-2.963133121515115) q[13];
ry(3.1091388554180965) q[14];
rz(-2.877548003263954) q[14];
ry(-1.5709090385212772) q[15];
rz(-2.9622156627137612) q[15];
ry(3.1384668445102313) q[16];
rz(2.0486807735072587) q[16];
ry(1.5703970676903785) q[17];
rz(-1.3922479843956113) q[17];
ry(1.5717698833612577) q[18];
rz(-1.340497550055081) q[18];
ry(-0.0010898242652732648) q[19];
rz(-1.7364995537962704) q[19];