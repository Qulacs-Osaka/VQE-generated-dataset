OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(1.5651007595035722) q[0];
ry(2.2838854555178365) q[1];
cx q[0],q[1];
ry(-0.6534422453308197) q[0];
ry(2.306966538323418) q[1];
cx q[0],q[1];
ry(-1.3943389387287253) q[2];
ry(-0.5711943929375423) q[3];
cx q[2],q[3];
ry(-0.9208557034110998) q[2];
ry(-1.9825117106600392) q[3];
cx q[2],q[3];
ry(3.136580400337928) q[4];
ry(-2.293395245982321) q[5];
cx q[4],q[5];
ry(-1.3551155997401363) q[4];
ry(-2.647106605161609) q[5];
cx q[4],q[5];
ry(0.6285360589099903) q[6];
ry(-1.936202220420964) q[7];
cx q[6],q[7];
ry(-0.9320668526348073) q[6];
ry(1.9410451842162244) q[7];
cx q[6],q[7];
ry(-0.8049495598512539) q[8];
ry(1.2194018342022555) q[9];
cx q[8],q[9];
ry(-0.6537528352566035) q[8];
ry(-0.9001051274157783) q[9];
cx q[8],q[9];
ry(-1.4339568355329193) q[10];
ry(-1.3171088740750028) q[11];
cx q[10],q[11];
ry(-2.461543726835593) q[10];
ry(-1.3647119534062568) q[11];
cx q[10],q[11];
ry(-2.877615936509609) q[0];
ry(0.35602764110996704) q[2];
cx q[0],q[2];
ry(-1.109533694876829) q[0];
ry(2.9015962684769945) q[2];
cx q[0],q[2];
ry(-2.6356538269126646) q[2];
ry(-1.3022390581407273) q[4];
cx q[2],q[4];
ry(-0.5425313745382828) q[2];
ry(1.7974252703698008) q[4];
cx q[2],q[4];
ry(-1.7997272744241233) q[4];
ry(1.4435848578892705) q[6];
cx q[4],q[6];
ry(1.9171041549928853) q[4];
ry(0.785919197578699) q[6];
cx q[4],q[6];
ry(2.584708435489695) q[6];
ry(-1.7176845323826997) q[8];
cx q[6],q[8];
ry(-1.395834533573182) q[6];
ry(1.6583033073861804) q[8];
cx q[6],q[8];
ry(-2.0562990914809207) q[8];
ry(-2.300346637834598) q[10];
cx q[8],q[10];
ry(2.559812112624527) q[8];
ry(-1.4436353721833264) q[10];
cx q[8],q[10];
ry(2.420410540679596) q[1];
ry(-2.2071748212879534) q[3];
cx q[1],q[3];
ry(-2.519951264286467) q[1];
ry(2.5055373250271082) q[3];
cx q[1],q[3];
ry(-3.107447434415196) q[3];
ry(-2.8499886416676246) q[5];
cx q[3],q[5];
ry(1.489646360193753) q[3];
ry(-0.8430019013764642) q[5];
cx q[3],q[5];
ry(-0.24991376572528173) q[5];
ry(0.5199434628216404) q[7];
cx q[5],q[7];
ry(1.852741249845244) q[5];
ry(-2.287191865658042) q[7];
cx q[5],q[7];
ry(-2.6180525771293364) q[7];
ry(-2.711384957276875) q[9];
cx q[7],q[9];
ry(-2.5033934718414903) q[7];
ry(0.4685768716366245) q[9];
cx q[7],q[9];
ry(2.603992062514927) q[9];
ry(-0.8453495067639238) q[11];
cx q[9],q[11];
ry(1.8882405557886512) q[9];
ry(-2.639505616203669) q[11];
cx q[9],q[11];
ry(-1.003858800805754) q[0];
ry(0.07561767586386912) q[1];
cx q[0],q[1];
ry(-1.9842641839209811) q[0];
ry(1.613986918132378) q[1];
cx q[0],q[1];
ry(-1.1030076155079973) q[2];
ry(-2.3092684560530956) q[3];
cx q[2],q[3];
ry(-2.054362404031562) q[2];
ry(1.8103418341295279) q[3];
cx q[2],q[3];
ry(3.0051079224089157) q[4];
ry(-1.1401920369461775) q[5];
cx q[4],q[5];
ry(-1.8045653278025018) q[4];
ry(-0.6261703387694002) q[5];
cx q[4],q[5];
ry(-2.364061938916225) q[6];
ry(1.4579209291408306) q[7];
cx q[6],q[7];
ry(-2.830137708157398) q[6];
ry(2.5202762403630747) q[7];
cx q[6],q[7];
ry(-1.4933832397333944) q[8];
ry(-0.8304430462243133) q[9];
cx q[8],q[9];
ry(0.20662928518150991) q[8];
ry(-2.227130707557481) q[9];
cx q[8],q[9];
ry(-0.9322717857403752) q[10];
ry(-2.1475731386084123) q[11];
cx q[10],q[11];
ry(1.4046387322434066) q[10];
ry(-2.381792701508365) q[11];
cx q[10],q[11];
ry(-2.5244382655611664) q[0];
ry(2.7936270270165564) q[2];
cx q[0],q[2];
ry(-1.8346415229120128) q[0];
ry(-0.8358632821151044) q[2];
cx q[0],q[2];
ry(-2.2420947042764277) q[2];
ry(2.385936673026184) q[4];
cx q[2],q[4];
ry(1.4553581189861091) q[2];
ry(-1.7552118385651774) q[4];
cx q[2],q[4];
ry(-2.6303452165587236) q[4];
ry(-1.4946802985637346) q[6];
cx q[4],q[6];
ry(1.1845801015809387) q[4];
ry(-0.7775408819274556) q[6];
cx q[4],q[6];
ry(-1.7600218591815375) q[6];
ry(0.19817246847615633) q[8];
cx q[6],q[8];
ry(-0.04790273396506589) q[6];
ry(2.5450220507541386) q[8];
cx q[6],q[8];
ry(1.8407982464144927) q[8];
ry(0.19780103940588664) q[10];
cx q[8],q[10];
ry(-2.3238771785605663) q[8];
ry(-1.8816836667651597) q[10];
cx q[8],q[10];
ry(-1.3526222777913866) q[1];
ry(-1.343154323847012) q[3];
cx q[1],q[3];
ry(-1.4482128659870086) q[1];
ry(-0.31397191231117655) q[3];
cx q[1],q[3];
ry(1.0261342652839693) q[3];
ry(-2.8511736201236544) q[5];
cx q[3],q[5];
ry(-2.7666144109357385) q[3];
ry(1.245019403105078) q[5];
cx q[3],q[5];
ry(-1.1733091611631083) q[5];
ry(-2.1518673206907035) q[7];
cx q[5],q[7];
ry(-2.7409760191334076) q[5];
ry(-2.473313815396771) q[7];
cx q[5],q[7];
ry(-3.1252842683220883) q[7];
ry(2.2290450731894373) q[9];
cx q[7],q[9];
ry(1.3743180231440357) q[7];
ry(-1.088743933402651) q[9];
cx q[7],q[9];
ry(-1.5533313364087107) q[9];
ry(-2.429456712014582) q[11];
cx q[9],q[11];
ry(-2.362214129929667) q[9];
ry(2.401378971189947) q[11];
cx q[9],q[11];
ry(-0.7208949505541901) q[0];
ry(-1.804651877246486) q[1];
cx q[0],q[1];
ry(2.00676228251842) q[0];
ry(-2.395123012987084) q[1];
cx q[0],q[1];
ry(-1.2389250883726015) q[2];
ry(-0.6814551015821905) q[3];
cx q[2],q[3];
ry(-2.4736644241529064) q[2];
ry(2.0475053774357668) q[3];
cx q[2],q[3];
ry(1.983764798119851) q[4];
ry(-0.8729115886185967) q[5];
cx q[4],q[5];
ry(-0.40911586982970677) q[4];
ry(-1.0967137949640355) q[5];
cx q[4],q[5];
ry(-1.452031069319605) q[6];
ry(2.6022890260015137) q[7];
cx q[6],q[7];
ry(-1.1631007221873402) q[6];
ry(2.651934593150452) q[7];
cx q[6],q[7];
ry(-1.08502293966332) q[8];
ry(-0.6278429544585719) q[9];
cx q[8],q[9];
ry(0.4076109695676875) q[8];
ry(1.1733428090002578) q[9];
cx q[8],q[9];
ry(0.9907210134933615) q[10];
ry(-2.4628450694558834) q[11];
cx q[10],q[11];
ry(0.3422732913804622) q[10];
ry(0.19427985259861294) q[11];
cx q[10],q[11];
ry(1.9369231936019835) q[0];
ry(0.8969995685518162) q[2];
cx q[0],q[2];
ry(1.9477188299003352) q[0];
ry(0.4922939437001178) q[2];
cx q[0],q[2];
ry(-2.67529769528306) q[2];
ry(-1.1229862725840585) q[4];
cx q[2],q[4];
ry(-0.36384859409744114) q[2];
ry(-0.886252057508524) q[4];
cx q[2],q[4];
ry(-1.7056275882508967) q[4];
ry(-0.4012683322283044) q[6];
cx q[4],q[6];
ry(1.199732686938119) q[4];
ry(2.229641467311062) q[6];
cx q[4],q[6];
ry(-0.9758390525821659) q[6];
ry(-2.9941416121062856) q[8];
cx q[6],q[8];
ry(-0.7258766341989222) q[6];
ry(0.9595163643000993) q[8];
cx q[6],q[8];
ry(-2.1291498878369737) q[8];
ry(-3.0013431644817623) q[10];
cx q[8],q[10];
ry(-0.8954828073146341) q[8];
ry(-0.6304197458681544) q[10];
cx q[8],q[10];
ry(0.7254473270225478) q[1];
ry(-1.3820588374390943) q[3];
cx q[1],q[3];
ry(-2.7940492116402544) q[1];
ry(2.1377872367610418) q[3];
cx q[1],q[3];
ry(2.417599113901627) q[3];
ry(-2.088146018783903) q[5];
cx q[3],q[5];
ry(-0.43954833239447894) q[3];
ry(1.0377062457900204) q[5];
cx q[3],q[5];
ry(1.2013564671574763) q[5];
ry(2.726705992493371) q[7];
cx q[5],q[7];
ry(2.260045104013732) q[5];
ry(1.3406460837411938) q[7];
cx q[5],q[7];
ry(1.5253521479350765) q[7];
ry(1.5507059641174699) q[9];
cx q[7],q[9];
ry(-2.302035067962481) q[7];
ry(2.587674903709925) q[9];
cx q[7],q[9];
ry(-2.678294955209905) q[9];
ry(0.8876505910950575) q[11];
cx q[9],q[11];
ry(-0.43894183843870666) q[9];
ry(0.7677788852743692) q[11];
cx q[9],q[11];
ry(2.0096985158135814) q[0];
ry(1.9533128793917156) q[1];
cx q[0],q[1];
ry(1.7426204961444902) q[0];
ry(-2.113782883772977) q[1];
cx q[0],q[1];
ry(0.1950613448596978) q[2];
ry(-1.4609776978726363) q[3];
cx q[2],q[3];
ry(0.7074935665607605) q[2];
ry(1.8476028671858558) q[3];
cx q[2],q[3];
ry(0.8069028345831941) q[4];
ry(1.295375025969041) q[5];
cx q[4],q[5];
ry(0.4643401362475341) q[4];
ry(2.2348242278114627) q[5];
cx q[4],q[5];
ry(-2.3332933183859277) q[6];
ry(-0.7432858605359439) q[7];
cx q[6],q[7];
ry(-2.7775121043006186) q[6];
ry(-1.6284969907837574) q[7];
cx q[6],q[7];
ry(3.1021553956443624) q[8];
ry(-0.06519180365418009) q[9];
cx q[8],q[9];
ry(0.3966316870213751) q[8];
ry(0.9969477046022656) q[9];
cx q[8],q[9];
ry(-1.9798016906633924) q[10];
ry(-2.7705866652361673) q[11];
cx q[10],q[11];
ry(-1.6338704613608463) q[10];
ry(-1.654657669993507) q[11];
cx q[10],q[11];
ry(-0.3405547620032827) q[0];
ry(-0.33451031641052625) q[2];
cx q[0],q[2];
ry(2.4599564106924383) q[0];
ry(-2.363332019241504) q[2];
cx q[0],q[2];
ry(-0.3796918010738332) q[2];
ry(-2.4174639491020486) q[4];
cx q[2],q[4];
ry(-1.4639854478034182) q[2];
ry(-2.2920612925425576) q[4];
cx q[2],q[4];
ry(-0.910417624272716) q[4];
ry(1.1358116424009594) q[6];
cx q[4],q[6];
ry(0.21829134176448584) q[4];
ry(-2.664080177924307) q[6];
cx q[4],q[6];
ry(2.673647889943448) q[6];
ry(-2.9019207873363366) q[8];
cx q[6],q[8];
ry(-2.916550069265926) q[6];
ry(1.5426846798168432) q[8];
cx q[6],q[8];
ry(2.0920320927759537) q[8];
ry(1.8329430086020517) q[10];
cx q[8],q[10];
ry(-1.0401312491415222) q[8];
ry(-2.447384000477519) q[10];
cx q[8],q[10];
ry(0.5655597796929628) q[1];
ry(2.5379575560393617) q[3];
cx q[1],q[3];
ry(-2.5587555571104947) q[1];
ry(-2.5625036232402656) q[3];
cx q[1],q[3];
ry(-2.9569870309017823) q[3];
ry(1.549729240448902) q[5];
cx q[3],q[5];
ry(-1.7053987236202888) q[3];
ry(-1.037904171263857) q[5];
cx q[3],q[5];
ry(-2.1307511195125826) q[5];
ry(2.8682388791405042) q[7];
cx q[5],q[7];
ry(1.0558346866211155) q[5];
ry(2.7102916626363345) q[7];
cx q[5],q[7];
ry(2.2463582272738942) q[7];
ry(-1.1882014910194822) q[9];
cx q[7],q[9];
ry(-0.5231978519840902) q[7];
ry(1.3415804625716723) q[9];
cx q[7],q[9];
ry(-0.6469373208977225) q[9];
ry(-1.9240452335556453) q[11];
cx q[9],q[11];
ry(-1.1171076171435699) q[9];
ry(-2.9138782584158442) q[11];
cx q[9],q[11];
ry(-0.19064689383247527) q[0];
ry(1.4768332638155108) q[1];
cx q[0],q[1];
ry(-2.3531134542742107) q[0];
ry(2.769271186957946) q[1];
cx q[0],q[1];
ry(-1.204637954836704) q[2];
ry(-1.8650169290140897) q[3];
cx q[2],q[3];
ry(-0.6216265756273732) q[2];
ry(-0.009172111400620686) q[3];
cx q[2],q[3];
ry(1.7006166538468337) q[4];
ry(0.9755132977304513) q[5];
cx q[4],q[5];
ry(-1.9563089809869725) q[4];
ry(-2.5023880305862254) q[5];
cx q[4],q[5];
ry(-1.650284133528153) q[6];
ry(-2.041750013775881) q[7];
cx q[6],q[7];
ry(-1.4281763533531686) q[6];
ry(-0.5931152051473045) q[7];
cx q[6],q[7];
ry(-1.8273997556451653) q[8];
ry(1.938294606508161) q[9];
cx q[8],q[9];
ry(-1.2452565978846115) q[8];
ry(0.3424337928543233) q[9];
cx q[8],q[9];
ry(2.686990084439536) q[10];
ry(1.3127808296842796) q[11];
cx q[10],q[11];
ry(-2.9073014624259326) q[10];
ry(2.1792477508281545) q[11];
cx q[10],q[11];
ry(-1.3175268079642803) q[0];
ry(-1.8941098087899382) q[2];
cx q[0],q[2];
ry(-0.8108247575305283) q[0];
ry(-1.4529231726499985) q[2];
cx q[0],q[2];
ry(-0.8338752895655916) q[2];
ry(-2.101906633981651) q[4];
cx q[2],q[4];
ry(-0.934844408632806) q[2];
ry(2.4566954036444395) q[4];
cx q[2],q[4];
ry(-2.989457845802992) q[4];
ry(-2.788525625141247) q[6];
cx q[4],q[6];
ry(1.9005377486361923) q[4];
ry(2.80898751353013) q[6];
cx q[4],q[6];
ry(1.3300174278097812) q[6];
ry(2.592679251560106) q[8];
cx q[6],q[8];
ry(0.1271216478109212) q[6];
ry(-2.045036476757453) q[8];
cx q[6],q[8];
ry(1.0587659468294977) q[8];
ry(1.021168148553746) q[10];
cx q[8],q[10];
ry(2.56683310136805) q[8];
ry(0.6353106361547547) q[10];
cx q[8],q[10];
ry(2.9169225828369783) q[1];
ry(3.0564166523204923) q[3];
cx q[1],q[3];
ry(2.116350093472426) q[1];
ry(-2.4718809480077204) q[3];
cx q[1],q[3];
ry(0.5754161309032184) q[3];
ry(0.7737435231111789) q[5];
cx q[3],q[5];
ry(2.279120282182724) q[3];
ry(1.3021383517381888) q[5];
cx q[3],q[5];
ry(2.9600458595715513) q[5];
ry(-2.0561701604770475) q[7];
cx q[5],q[7];
ry(0.28880385731716357) q[5];
ry(2.3980828767738727) q[7];
cx q[5],q[7];
ry(-0.7065676197082533) q[7];
ry(0.4100605423054568) q[9];
cx q[7],q[9];
ry(-2.694377646110151) q[7];
ry(-2.5016063605278553) q[9];
cx q[7],q[9];
ry(-0.9079770438076696) q[9];
ry(2.887636066693205) q[11];
cx q[9],q[11];
ry(-0.33921295321564954) q[9];
ry(-2.8688625166201485) q[11];
cx q[9],q[11];
ry(1.4750441325743928) q[0];
ry(2.4992888857436704) q[1];
cx q[0],q[1];
ry(1.1953231832273419) q[0];
ry(-0.6340474509507185) q[1];
cx q[0],q[1];
ry(-2.481609897678557) q[2];
ry(1.8993070663945115) q[3];
cx q[2],q[3];
ry(-1.750761020789643) q[2];
ry(-0.9317795847429161) q[3];
cx q[2],q[3];
ry(-1.9503755772577132) q[4];
ry(-0.5853622652966317) q[5];
cx q[4],q[5];
ry(-0.6306852768317945) q[4];
ry(-1.452391656749051) q[5];
cx q[4],q[5];
ry(-1.0632556533906272) q[6];
ry(0.9109146168798565) q[7];
cx q[6],q[7];
ry(3.04993996515383) q[6];
ry(0.8290538957533728) q[7];
cx q[6],q[7];
ry(1.04871819770311) q[8];
ry(-0.36478493968842907) q[9];
cx q[8],q[9];
ry(-1.1399141126557504) q[8];
ry(0.2148261674538691) q[9];
cx q[8],q[9];
ry(-2.8288536216579336) q[10];
ry(-0.9683477559719469) q[11];
cx q[10],q[11];
ry(-2.2243869270151926) q[10];
ry(1.2368851537796663) q[11];
cx q[10],q[11];
ry(2.7326352840926713) q[0];
ry(-2.359933682344136) q[2];
cx q[0],q[2];
ry(1.5975791180204966) q[0];
ry(2.4960716552807667) q[2];
cx q[0],q[2];
ry(-2.5839609335316727) q[2];
ry(1.8800786042753685) q[4];
cx q[2],q[4];
ry(-1.3429848535293196) q[2];
ry(-0.3984316513537451) q[4];
cx q[2],q[4];
ry(2.3359928237065715) q[4];
ry(2.271248185658661) q[6];
cx q[4],q[6];
ry(2.8094933200153016) q[4];
ry(2.0870278622446494) q[6];
cx q[4],q[6];
ry(1.8183656815679576) q[6];
ry(1.1108663170143727) q[8];
cx q[6],q[8];
ry(2.87183518851166) q[6];
ry(-2.4513245146217857) q[8];
cx q[6],q[8];
ry(-1.8769670333513169) q[8];
ry(-0.38577099073253507) q[10];
cx q[8],q[10];
ry(1.5333393681294094) q[8];
ry(0.988496632117543) q[10];
cx q[8],q[10];
ry(1.7981037748909872) q[1];
ry(-2.4308054769743173) q[3];
cx q[1],q[3];
ry(-3.010096186050777) q[1];
ry(2.725697966062101) q[3];
cx q[1],q[3];
ry(-3.0243934412756412) q[3];
ry(2.334723969091574) q[5];
cx q[3],q[5];
ry(-1.509857994291024) q[3];
ry(-1.3522612432899308) q[5];
cx q[3],q[5];
ry(-0.8388890204545288) q[5];
ry(-0.8636423130093107) q[7];
cx q[5],q[7];
ry(-0.7805909749018091) q[5];
ry(-2.411919315445145) q[7];
cx q[5],q[7];
ry(-1.1317123095752777) q[7];
ry(-1.1290071881373507) q[9];
cx q[7],q[9];
ry(-0.7643622789396112) q[7];
ry(-0.3370956913573488) q[9];
cx q[7],q[9];
ry(2.2080006457094443) q[9];
ry(-2.966338797909029) q[11];
cx q[9],q[11];
ry(1.1484125272399222) q[9];
ry(-1.8229690798201972) q[11];
cx q[9],q[11];
ry(0.6104687004125031) q[0];
ry(0.10282063326101554) q[1];
cx q[0],q[1];
ry(2.79698439288864) q[0];
ry(1.1481999520175936) q[1];
cx q[0],q[1];
ry(0.8482163008467856) q[2];
ry(0.9268471901040609) q[3];
cx q[2],q[3];
ry(-1.0013660596751572) q[2];
ry(2.033799925015638) q[3];
cx q[2],q[3];
ry(-1.1561112020642481) q[4];
ry(2.1432882207261934) q[5];
cx q[4],q[5];
ry(0.3184791586081286) q[4];
ry(-1.893264725062856) q[5];
cx q[4],q[5];
ry(-2.126269709433795) q[6];
ry(2.043423648357561) q[7];
cx q[6],q[7];
ry(-0.977992149655249) q[6];
ry(-1.021973967368973) q[7];
cx q[6],q[7];
ry(-0.06701624891444968) q[8];
ry(0.47440250627904756) q[9];
cx q[8],q[9];
ry(-2.2073234617564235) q[8];
ry(2.1229580753656956) q[9];
cx q[8],q[9];
ry(0.021062837319044014) q[10];
ry(1.2000271221871186) q[11];
cx q[10],q[11];
ry(-1.6913308547050194) q[10];
ry(2.7895718590556466) q[11];
cx q[10],q[11];
ry(-0.6360257072163824) q[0];
ry(-2.005022007486466) q[2];
cx q[0],q[2];
ry(0.8871413914333065) q[0];
ry(-0.02049220808139829) q[2];
cx q[0],q[2];
ry(2.5969277359891594) q[2];
ry(-2.0609640050170204) q[4];
cx q[2],q[4];
ry(1.6504362242694413) q[2];
ry(0.9452449705625039) q[4];
cx q[2],q[4];
ry(2.0001368268917377) q[4];
ry(-0.03869365779525204) q[6];
cx q[4],q[6];
ry(-1.668110399832247) q[4];
ry(0.18407115329751783) q[6];
cx q[4],q[6];
ry(0.45239910516605253) q[6];
ry(2.408814931239731) q[8];
cx q[6],q[8];
ry(2.899549618192423) q[6];
ry(1.3613430391358259) q[8];
cx q[6],q[8];
ry(-2.3095411103231593) q[8];
ry(-2.8818453420607266) q[10];
cx q[8],q[10];
ry(-2.1489955468086714) q[8];
ry(1.9565564325104872) q[10];
cx q[8],q[10];
ry(2.6142643636373757) q[1];
ry(1.502086814752877) q[3];
cx q[1],q[3];
ry(-0.5222612587631987) q[1];
ry(1.062132394894677) q[3];
cx q[1],q[3];
ry(-0.36227119901157645) q[3];
ry(-1.498619581988151) q[5];
cx q[3],q[5];
ry(-1.0873995179619804) q[3];
ry(2.0048753289472576) q[5];
cx q[3],q[5];
ry(0.6164837423828209) q[5];
ry(3.08494518620389) q[7];
cx q[5],q[7];
ry(-2.7243158482970316) q[5];
ry(2.0696741746505447) q[7];
cx q[5],q[7];
ry(0.6111975505985949) q[7];
ry(1.7481509201322425) q[9];
cx q[7],q[9];
ry(2.214882216470392) q[7];
ry(-1.6066509007143017) q[9];
cx q[7],q[9];
ry(-1.8331349140052344) q[9];
ry(0.15277471478190652) q[11];
cx q[9],q[11];
ry(2.9637125840013767) q[9];
ry(1.960679225972001) q[11];
cx q[9],q[11];
ry(-1.5600969333541639) q[0];
ry(-1.7274414709811892) q[1];
cx q[0],q[1];
ry(2.2377852458314096) q[0];
ry(0.6484213373063445) q[1];
cx q[0],q[1];
ry(-2.508557352734878) q[2];
ry(-1.078501990725517) q[3];
cx q[2],q[3];
ry(1.1201507535285868) q[2];
ry(3.034154529188842) q[3];
cx q[2],q[3];
ry(-2.524793007758989) q[4];
ry(-2.012283986150024) q[5];
cx q[4],q[5];
ry(-2.2360579364780717) q[4];
ry(0.19664383853956302) q[5];
cx q[4],q[5];
ry(-0.05179534779348994) q[6];
ry(-2.7025473311479424) q[7];
cx q[6],q[7];
ry(-2.896298624382663) q[6];
ry(-0.23143222712215256) q[7];
cx q[6],q[7];
ry(0.4092400526752406) q[8];
ry(-1.7717030090591823) q[9];
cx q[8],q[9];
ry(0.5827854584878924) q[8];
ry(2.7212533299400774) q[9];
cx q[8],q[9];
ry(-2.4804097467325206) q[10];
ry(-2.4207926899008285) q[11];
cx q[10],q[11];
ry(2.835453032951318) q[10];
ry(0.6290175388961706) q[11];
cx q[10],q[11];
ry(3.0514248813313793) q[0];
ry(-1.7554295197347085) q[2];
cx q[0],q[2];
ry(-0.2150507872890124) q[0];
ry(2.936026473593915) q[2];
cx q[0],q[2];
ry(0.5249377519618369) q[2];
ry(-2.207576181786011) q[4];
cx q[2],q[4];
ry(-2.2150016343973524) q[2];
ry(1.3712514475672286) q[4];
cx q[2],q[4];
ry(-1.9928572086858962) q[4];
ry(0.05411216438689218) q[6];
cx q[4],q[6];
ry(2.2443525263047572) q[4];
ry(-0.7824361481989097) q[6];
cx q[4],q[6];
ry(1.831626251152695) q[6];
ry(0.5818308272668021) q[8];
cx q[6],q[8];
ry(-1.2125648492410814) q[6];
ry(2.618509596301485) q[8];
cx q[6],q[8];
ry(0.7133570097149755) q[8];
ry(-3.055625885071591) q[10];
cx q[8],q[10];
ry(0.39446340458716467) q[8];
ry(-1.3407837870936585) q[10];
cx q[8],q[10];
ry(2.7088009384292873) q[1];
ry(-2.052555165058534) q[3];
cx q[1],q[3];
ry(-1.1106719304423063) q[1];
ry(-2.5797107557075676) q[3];
cx q[1],q[3];
ry(-2.713317200856053) q[3];
ry(-1.0639681639001104) q[5];
cx q[3],q[5];
ry(2.627791376753288) q[3];
ry(0.48954267275007096) q[5];
cx q[3],q[5];
ry(-2.2414579336017235) q[5];
ry(1.1701842770386641) q[7];
cx q[5],q[7];
ry(-2.7383725934961323) q[5];
ry(-2.399670475792111) q[7];
cx q[5],q[7];
ry(2.0511730391693765) q[7];
ry(-1.1626987133024969) q[9];
cx q[7],q[9];
ry(-2.2961346199543904) q[7];
ry(-0.7439456655076295) q[9];
cx q[7],q[9];
ry(0.05050686682818369) q[9];
ry(-2.5485169195791433) q[11];
cx q[9],q[11];
ry(2.995861969101537) q[9];
ry(-2.9846148179854115) q[11];
cx q[9],q[11];
ry(0.08621271235218497) q[0];
ry(-2.594713490347546) q[1];
cx q[0],q[1];
ry(2.2406085058627276) q[0];
ry(-0.5315807569627733) q[1];
cx q[0],q[1];
ry(-2.2970657833988413) q[2];
ry(1.8974828100854495) q[3];
cx q[2],q[3];
ry(-0.6351572650874742) q[2];
ry(-2.4634353328796967) q[3];
cx q[2],q[3];
ry(-3.0615381222433093) q[4];
ry(1.5612631556369283) q[5];
cx q[4],q[5];
ry(-1.0565886855497901) q[4];
ry(-2.585643197535889) q[5];
cx q[4],q[5];
ry(-2.4982841580334387) q[6];
ry(3.0199081589968486) q[7];
cx q[6],q[7];
ry(0.7890081632783108) q[6];
ry(0.7259107470610723) q[7];
cx q[6],q[7];
ry(2.908774774850981) q[8];
ry(-2.5225379717793235) q[9];
cx q[8],q[9];
ry(0.7644380907354673) q[8];
ry(2.1553903332512476) q[9];
cx q[8],q[9];
ry(2.3640886259218012) q[10];
ry(-0.4136026101774428) q[11];
cx q[10],q[11];
ry(1.1079830654981302) q[10];
ry(1.4601219313075857) q[11];
cx q[10],q[11];
ry(-2.723747633218609) q[0];
ry(1.2657820388752556) q[2];
cx q[0],q[2];
ry(1.456831561722555) q[0];
ry(0.484116833050925) q[2];
cx q[0],q[2];
ry(1.7703801362343605) q[2];
ry(1.8760758524870809) q[4];
cx q[2],q[4];
ry(-1.7002028243036273) q[2];
ry(-1.105677846160785) q[4];
cx q[2],q[4];
ry(1.7469327552243006) q[4];
ry(-2.009789376196542) q[6];
cx q[4],q[6];
ry(-1.6986449232937395) q[4];
ry(1.2535752349637648) q[6];
cx q[4],q[6];
ry(0.5478981223125503) q[6];
ry(-2.670313485127196) q[8];
cx q[6],q[8];
ry(-2.2632453840823117) q[6];
ry(2.425116909220934) q[8];
cx q[6],q[8];
ry(-2.006023079068499) q[8];
ry(0.7038274991317801) q[10];
cx q[8],q[10];
ry(-0.27035385041373416) q[8];
ry(-0.8353982955524545) q[10];
cx q[8],q[10];
ry(-0.7686469666059932) q[1];
ry(2.457196065872897) q[3];
cx q[1],q[3];
ry(2.661249822559024) q[1];
ry(0.8376923597115526) q[3];
cx q[1],q[3];
ry(-2.0759351585973613) q[3];
ry(1.039084535947551) q[5];
cx q[3],q[5];
ry(-0.8836438235031108) q[3];
ry(-2.5354948593369597) q[5];
cx q[3],q[5];
ry(-1.8257429512311694) q[5];
ry(-0.1974771181266703) q[7];
cx q[5],q[7];
ry(2.9715278140756016) q[5];
ry(2.0723308721011895) q[7];
cx q[5],q[7];
ry(1.9709217321578087) q[7];
ry(0.569115207415415) q[9];
cx q[7],q[9];
ry(1.865204925031929) q[7];
ry(-3.0754485204899575) q[9];
cx q[7],q[9];
ry(-3.0526903980852462) q[9];
ry(0.05088914969144711) q[11];
cx q[9],q[11];
ry(-0.2569205543515949) q[9];
ry(-1.3174295072709232) q[11];
cx q[9],q[11];
ry(-0.6754687668240311) q[0];
ry(-1.1142018521338404) q[1];
cx q[0],q[1];
ry(-1.5855484098505523) q[0];
ry(0.6988107299645365) q[1];
cx q[0],q[1];
ry(-0.3079679878115329) q[2];
ry(1.9141267614140123) q[3];
cx q[2],q[3];
ry(-3.0200353759233756) q[2];
ry(-1.6815467279207494) q[3];
cx q[2],q[3];
ry(2.4939098109734816) q[4];
ry(2.5128276177240805) q[5];
cx q[4],q[5];
ry(-1.2266379022813618) q[4];
ry(1.85346908102116) q[5];
cx q[4],q[5];
ry(-1.2951687013823554) q[6];
ry(2.4997122308286417) q[7];
cx q[6],q[7];
ry(1.7688345989684755) q[6];
ry(0.3882876409160618) q[7];
cx q[6],q[7];
ry(-0.5760423272608639) q[8];
ry(-1.473213110554163) q[9];
cx q[8],q[9];
ry(-0.6745511643596922) q[8];
ry(2.2313346465703043) q[9];
cx q[8],q[9];
ry(-2.9030004303508714) q[10];
ry(2.4992593915558245) q[11];
cx q[10],q[11];
ry(-2.88502415071374) q[10];
ry(-0.6009932527137153) q[11];
cx q[10],q[11];
ry(-2.35059211227675) q[0];
ry(2.3991824532770663) q[2];
cx q[0],q[2];
ry(-2.2715515299106377) q[0];
ry(2.7263018348588686) q[2];
cx q[0],q[2];
ry(-2.5407936235881086) q[2];
ry(2.4133894324545073) q[4];
cx q[2],q[4];
ry(-2.28539008713506) q[2];
ry(-1.164683401947335) q[4];
cx q[2],q[4];
ry(-0.015566163941824307) q[4];
ry(2.380484853331596) q[6];
cx q[4],q[6];
ry(-2.1522583097230683) q[4];
ry(1.8452695485535948) q[6];
cx q[4],q[6];
ry(-2.66120924641117) q[6];
ry(0.3434686167554073) q[8];
cx q[6],q[8];
ry(-1.2176896439772333) q[6];
ry(0.2740601612158575) q[8];
cx q[6],q[8];
ry(-0.14107175200157496) q[8];
ry(-2.4616748281522622) q[10];
cx q[8],q[10];
ry(0.505676340597697) q[8];
ry(1.0593700123140426) q[10];
cx q[8],q[10];
ry(-1.109877003017238) q[1];
ry(-0.6366588387810223) q[3];
cx q[1],q[3];
ry(2.4911881417723185) q[1];
ry(2.6829450445653062) q[3];
cx q[1],q[3];
ry(1.6469352358484044) q[3];
ry(-0.8782811708284122) q[5];
cx q[3],q[5];
ry(-2.063032955150253) q[3];
ry(-0.6903891244428716) q[5];
cx q[3],q[5];
ry(3.130858387180525) q[5];
ry(1.0445450146231927) q[7];
cx q[5],q[7];
ry(1.3594001721806872) q[5];
ry(2.6577361076930113) q[7];
cx q[5],q[7];
ry(1.7171178266403855) q[7];
ry(1.9137329947538022) q[9];
cx q[7],q[9];
ry(2.647228199739309) q[7];
ry(-0.5236553616919887) q[9];
cx q[7],q[9];
ry(2.57242204859096) q[9];
ry(-0.07390236956571616) q[11];
cx q[9],q[11];
ry(-1.0705490452607644) q[9];
ry(-1.6233918939420828) q[11];
cx q[9],q[11];
ry(-1.83245288397708) q[0];
ry(1.2824965852692722) q[1];
cx q[0],q[1];
ry(-1.0345693591829237) q[0];
ry(0.20028587501317488) q[1];
cx q[0],q[1];
ry(-1.0289621135695652) q[2];
ry(-0.23877117850054475) q[3];
cx q[2],q[3];
ry(2.3379539261208016) q[2];
ry(0.3938800784384735) q[3];
cx q[2],q[3];
ry(-1.6493087079271929) q[4];
ry(0.2428739569783538) q[5];
cx q[4],q[5];
ry(-1.2232284721823605) q[4];
ry(-2.548158896418964) q[5];
cx q[4],q[5];
ry(1.5292500762523957) q[6];
ry(-2.486682768551738) q[7];
cx q[6],q[7];
ry(-1.7655860918642006) q[6];
ry(0.6231829492151494) q[7];
cx q[6],q[7];
ry(0.5569964101329408) q[8];
ry(2.525395016794532) q[9];
cx q[8],q[9];
ry(2.766588637165514) q[8];
ry(1.5948373700745275) q[9];
cx q[8],q[9];
ry(-1.2278678462823047) q[10];
ry(2.34807937063475) q[11];
cx q[10],q[11];
ry(1.6731006125960493) q[10];
ry(-1.831922849587838) q[11];
cx q[10],q[11];
ry(-2.054865772271409) q[0];
ry(2.7728967967362603) q[2];
cx q[0],q[2];
ry(-2.4465517240120174) q[0];
ry(-0.45572194674110467) q[2];
cx q[0],q[2];
ry(2.812531638856697) q[2];
ry(2.8507405869292226) q[4];
cx q[2],q[4];
ry(-2.686145344883456) q[2];
ry(2.4395228460011626) q[4];
cx q[2],q[4];
ry(0.5426211737562378) q[4];
ry(-0.3216781390081264) q[6];
cx q[4],q[6];
ry(-1.8652977226428125) q[4];
ry(2.8266996876540165) q[6];
cx q[4],q[6];
ry(2.8334026522464244) q[6];
ry(-1.9034689730885754) q[8];
cx q[6],q[8];
ry(0.28898335562908084) q[6];
ry(0.3840824569774197) q[8];
cx q[6],q[8];
ry(0.6591108948545124) q[8];
ry(-0.19373951229512343) q[10];
cx q[8],q[10];
ry(2.7034676675372804) q[8];
ry(-1.6364264042668477) q[10];
cx q[8],q[10];
ry(0.2675818213057699) q[1];
ry(0.06928683048086359) q[3];
cx q[1],q[3];
ry(2.8486434133559833) q[1];
ry(2.035901354737824) q[3];
cx q[1],q[3];
ry(0.1710253578112333) q[3];
ry(-1.7051884971894495) q[5];
cx q[3],q[5];
ry(-1.9584213430449111) q[3];
ry(-0.9285420016640352) q[5];
cx q[3],q[5];
ry(-1.3729500139762585) q[5];
ry(-1.0773019637467955) q[7];
cx q[5],q[7];
ry(1.1652553259275855) q[5];
ry(-0.7160719400178793) q[7];
cx q[5],q[7];
ry(-2.287886017488289) q[7];
ry(-1.9914003882746716) q[9];
cx q[7],q[9];
ry(-2.3001975411131035) q[7];
ry(2.5825888363854492) q[9];
cx q[7],q[9];
ry(-1.5661856597543802) q[9];
ry(-0.08356237841543734) q[11];
cx q[9],q[11];
ry(-1.5960853241419748) q[9];
ry(1.9325494130073455) q[11];
cx q[9],q[11];
ry(-2.8515796175026744) q[0];
ry(1.7360507040149045) q[1];
cx q[0],q[1];
ry(0.3849583452068597) q[0];
ry(0.536989287111632) q[1];
cx q[0],q[1];
ry(1.774691539717355) q[2];
ry(-0.16420137981606117) q[3];
cx q[2],q[3];
ry(2.3179730986609877) q[2];
ry(1.4947388573898444) q[3];
cx q[2],q[3];
ry(2.614956415054728) q[4];
ry(0.698979098502698) q[5];
cx q[4],q[5];
ry(-1.6228144977290302) q[4];
ry(2.2003841070209234) q[5];
cx q[4],q[5];
ry(2.0503868453426115) q[6];
ry(-2.0281026398203723) q[7];
cx q[6],q[7];
ry(0.35343825050730293) q[6];
ry(-1.8899362588101807) q[7];
cx q[6],q[7];
ry(1.440698423642917) q[8];
ry(1.7579270613526234) q[9];
cx q[8],q[9];
ry(2.66383733830005) q[8];
ry(-2.9549174556299533) q[9];
cx q[8],q[9];
ry(2.450730879493764) q[10];
ry(-2.5738758145530674) q[11];
cx q[10],q[11];
ry(1.8707293958956728) q[10];
ry(-1.511828061420217) q[11];
cx q[10],q[11];
ry(-0.8837187115406175) q[0];
ry(0.512178221089477) q[2];
cx q[0],q[2];
ry(2.3629922798244913) q[0];
ry(-2.7523453827804265) q[2];
cx q[0],q[2];
ry(-0.9314228232625386) q[2];
ry(-2.3232356876794102) q[4];
cx q[2],q[4];
ry(1.248146856240989) q[2];
ry(-0.7035149798236064) q[4];
cx q[2],q[4];
ry(-2.2995939447873193) q[4];
ry(2.5506449706635097) q[6];
cx q[4],q[6];
ry(-1.9357071373971682) q[4];
ry(-0.14377707065650913) q[6];
cx q[4],q[6];
ry(2.06728154880763) q[6];
ry(1.5642228686115005) q[8];
cx q[6],q[8];
ry(0.38261228077878573) q[6];
ry(0.7537106910793208) q[8];
cx q[6],q[8];
ry(-1.6037026927431466) q[8];
ry(1.3331959464512886) q[10];
cx q[8],q[10];
ry(0.7383927351857631) q[8];
ry(-2.244735476411004) q[10];
cx q[8],q[10];
ry(-0.6479637415848485) q[1];
ry(-1.6144145609250826) q[3];
cx q[1],q[3];
ry(2.3127191403459166) q[1];
ry(-2.844755528839423) q[3];
cx q[1],q[3];
ry(-2.7047754987367787) q[3];
ry(1.8539816062290182) q[5];
cx q[3],q[5];
ry(0.2523919552916567) q[3];
ry(-1.8704630104699371) q[5];
cx q[3],q[5];
ry(2.558829736350582) q[5];
ry(-1.9633552575942819) q[7];
cx q[5],q[7];
ry(2.1046349643216398) q[5];
ry(1.519137674774052) q[7];
cx q[5],q[7];
ry(2.979017780852434) q[7];
ry(0.853216062766648) q[9];
cx q[7],q[9];
ry(-2.2024542137051144) q[7];
ry(1.4895912328230048) q[9];
cx q[7],q[9];
ry(0.6717791439677637) q[9];
ry(2.7357376897485897) q[11];
cx q[9],q[11];
ry(0.5140917769777085) q[9];
ry(2.514350595968694) q[11];
cx q[9],q[11];
ry(1.2412026001567298) q[0];
ry(-0.05104911704685785) q[1];
cx q[0],q[1];
ry(1.1448730281362416) q[0];
ry(0.06939529634310926) q[1];
cx q[0],q[1];
ry(0.35466584818396224) q[2];
ry(-2.5273132742893236) q[3];
cx q[2],q[3];
ry(1.3121746829567877) q[2];
ry(-1.7846372685643133) q[3];
cx q[2],q[3];
ry(-0.1845461431874007) q[4];
ry(-1.938538683376101) q[5];
cx q[4],q[5];
ry(0.20609343316609952) q[4];
ry(-1.2290170924860453) q[5];
cx q[4],q[5];
ry(0.3115104867901332) q[6];
ry(3.1389592790703515) q[7];
cx q[6],q[7];
ry(0.501011222051738) q[6];
ry(-0.9886253488945348) q[7];
cx q[6],q[7];
ry(-3.057678195129929) q[8];
ry(-0.07214838604183893) q[9];
cx q[8],q[9];
ry(-1.2408597953884755) q[8];
ry(-0.5993389145704948) q[9];
cx q[8],q[9];
ry(2.930168991431533) q[10];
ry(3.005653890086793) q[11];
cx q[10],q[11];
ry(0.8380564069344018) q[10];
ry(2.255321375386247) q[11];
cx q[10],q[11];
ry(1.0467393314453364) q[0];
ry(1.1453686833108758) q[2];
cx q[0],q[2];
ry(-2.076363959927832) q[0];
ry(-2.4635839087579128) q[2];
cx q[0],q[2];
ry(1.5611336397662248) q[2];
ry(2.8641786037291066) q[4];
cx q[2],q[4];
ry(2.856551692291084) q[2];
ry(0.7489564182066096) q[4];
cx q[2],q[4];
ry(1.3029589518264988) q[4];
ry(2.917281371220564) q[6];
cx q[4],q[6];
ry(-2.459683375696684) q[4];
ry(-0.8924269927364199) q[6];
cx q[4],q[6];
ry(-1.5293434733627447) q[6];
ry(-1.3311093653277029) q[8];
cx q[6],q[8];
ry(2.086875497772905) q[6];
ry(-0.713122297280173) q[8];
cx q[6],q[8];
ry(0.4055309208310849) q[8];
ry(0.8217998369977205) q[10];
cx q[8],q[10];
ry(0.9797249068443596) q[8];
ry(2.417035467002999) q[10];
cx q[8],q[10];
ry(-1.0294694237580782) q[1];
ry(-2.5587024807266823) q[3];
cx q[1],q[3];
ry(2.150036859511249) q[1];
ry(2.687534087588675) q[3];
cx q[1],q[3];
ry(-1.0560733453142424) q[3];
ry(-1.6003056475090718) q[5];
cx q[3],q[5];
ry(0.9040424111401522) q[3];
ry(-2.0706705071721894) q[5];
cx q[3],q[5];
ry(1.9165805642818512) q[5];
ry(2.1435585511281148) q[7];
cx q[5],q[7];
ry(0.9002760139982726) q[5];
ry(-1.008918716769434) q[7];
cx q[5],q[7];
ry(-2.0093802643870093) q[7];
ry(2.9657574184032813) q[9];
cx q[7],q[9];
ry(-2.0630980749544685) q[7];
ry(0.5326894999648283) q[9];
cx q[7],q[9];
ry(2.7684124586596037) q[9];
ry(-1.173298604065596) q[11];
cx q[9],q[11];
ry(0.7166944255317373) q[9];
ry(-1.7459850579587057) q[11];
cx q[9],q[11];
ry(-3.1109642452908517) q[0];
ry(0.8808162299852098) q[1];
cx q[0],q[1];
ry(-0.4599450545229375) q[0];
ry(-1.0431908715636833) q[1];
cx q[0],q[1];
ry(-2.8142464771553515) q[2];
ry(2.5241192671212573) q[3];
cx q[2],q[3];
ry(-0.27509538141848555) q[2];
ry(-1.5604947128734745) q[3];
cx q[2],q[3];
ry(0.16755283327950288) q[4];
ry(0.6112048823729896) q[5];
cx q[4],q[5];
ry(-0.22487573465991043) q[4];
ry(0.32618500875590895) q[5];
cx q[4],q[5];
ry(1.0514130407835023) q[6];
ry(-1.8287746195903913) q[7];
cx q[6],q[7];
ry(2.1875404276063795) q[6];
ry(-0.9646359419409878) q[7];
cx q[6],q[7];
ry(1.3089893822990897) q[8];
ry(-2.5598972002170326) q[9];
cx q[8],q[9];
ry(-2.660035779994986) q[8];
ry(-2.5360785906392915) q[9];
cx q[8],q[9];
ry(-1.3432155679659508) q[10];
ry(-0.9630055045111698) q[11];
cx q[10],q[11];
ry(-1.1239521642479473) q[10];
ry(-2.4962068905899675) q[11];
cx q[10],q[11];
ry(0.2771082988084178) q[0];
ry(-2.464657261488453) q[2];
cx q[0],q[2];
ry(-2.4493099296694276) q[0];
ry(0.2865964365837774) q[2];
cx q[0],q[2];
ry(-2.9499136521262) q[2];
ry(1.649886086030302) q[4];
cx q[2],q[4];
ry(-1.3367615157067922) q[2];
ry(0.7352698481022694) q[4];
cx q[2],q[4];
ry(2.290467703706166) q[4];
ry(0.8848966542266663) q[6];
cx q[4],q[6];
ry(0.6468539551676755) q[4];
ry(-0.3971136634345186) q[6];
cx q[4],q[6];
ry(-2.725700587646194) q[6];
ry(2.562021922346607) q[8];
cx q[6],q[8];
ry(2.006744986260813) q[6];
ry(0.9446011491169324) q[8];
cx q[6],q[8];
ry(-1.3748669085075162) q[8];
ry(2.0771484068151853) q[10];
cx q[8],q[10];
ry(2.525209560884362) q[8];
ry(0.894362941159703) q[10];
cx q[8],q[10];
ry(-2.0447829911263065) q[1];
ry(-2.9073068578262977) q[3];
cx q[1],q[3];
ry(-2.97470250716621) q[1];
ry(1.7078785430923147) q[3];
cx q[1],q[3];
ry(1.2810230858127776) q[3];
ry(-1.599028606552829) q[5];
cx q[3],q[5];
ry(0.7242981811120046) q[3];
ry(2.1788316284382296) q[5];
cx q[3],q[5];
ry(-0.36847088305651904) q[5];
ry(-1.189622951427908) q[7];
cx q[5],q[7];
ry(2.586090844224059) q[5];
ry(0.47222014777495863) q[7];
cx q[5],q[7];
ry(3.0140760199190995) q[7];
ry(-0.09092129775946799) q[9];
cx q[7],q[9];
ry(1.9702481793692428) q[7];
ry(-2.7693444828752303) q[9];
cx q[7],q[9];
ry(-1.5884430697226346) q[9];
ry(-0.8197074939540858) q[11];
cx q[9],q[11];
ry(2.175453351850158) q[9];
ry(1.6165619450583881) q[11];
cx q[9],q[11];
ry(-2.6980744252432176) q[0];
ry(1.5559004165690116) q[1];
cx q[0],q[1];
ry(2.0682288758536735) q[0];
ry(-2.494605539048435) q[1];
cx q[0],q[1];
ry(2.35584082449646) q[2];
ry(0.9444704476182872) q[3];
cx q[2],q[3];
ry(-0.5517302041058771) q[2];
ry(-2.5899536572127544) q[3];
cx q[2],q[3];
ry(-0.1400395872488268) q[4];
ry(-2.636338989542929) q[5];
cx q[4],q[5];
ry(1.5826961786599272) q[4];
ry(-0.7793505195363863) q[5];
cx q[4],q[5];
ry(1.9754840075888112) q[6];
ry(-2.0373002831245977) q[7];
cx q[6],q[7];
ry(-0.5379095974107169) q[6];
ry(0.41489271543738315) q[7];
cx q[6],q[7];
ry(-0.8627873624457871) q[8];
ry(-0.2206471492411885) q[9];
cx q[8],q[9];
ry(-1.1138175416322997) q[8];
ry(-0.7968996170953133) q[9];
cx q[8],q[9];
ry(0.9568484479275153) q[10];
ry(2.1061174928911184) q[11];
cx q[10],q[11];
ry(1.079320664585322) q[10];
ry(1.7210649164198752) q[11];
cx q[10],q[11];
ry(0.1985882769565697) q[0];
ry(-2.181177415111264) q[2];
cx q[0],q[2];
ry(1.1565004764224183) q[0];
ry(1.3267887154261766) q[2];
cx q[0],q[2];
ry(-0.7563241613400278) q[2];
ry(-1.9256409319475647) q[4];
cx q[2],q[4];
ry(0.19553820678392286) q[2];
ry(1.1287221756910844) q[4];
cx q[2],q[4];
ry(-0.8296501259635898) q[4];
ry(-2.5288985408972695) q[6];
cx q[4],q[6];
ry(-1.9200236779324729) q[4];
ry(-2.0516485596683776) q[6];
cx q[4],q[6];
ry(1.0410453070845413) q[6];
ry(2.4882043243733167) q[8];
cx q[6],q[8];
ry(1.0216977308778095) q[6];
ry(1.6440057859788495) q[8];
cx q[6],q[8];
ry(0.4175022802498187) q[8];
ry(2.824487084310511) q[10];
cx q[8],q[10];
ry(1.2969730898033238) q[8];
ry(-2.475168672168854) q[10];
cx q[8],q[10];
ry(1.3340860043398877) q[1];
ry(0.37251593305124153) q[3];
cx q[1],q[3];
ry(-2.9549865764000454) q[1];
ry(-0.729432258974608) q[3];
cx q[1],q[3];
ry(3.048501834249879) q[3];
ry(0.9663511300465424) q[5];
cx q[3],q[5];
ry(-2.5836750960847827) q[3];
ry(-0.5950371524735422) q[5];
cx q[3],q[5];
ry(-0.5378968857986957) q[5];
ry(-2.983355045006022) q[7];
cx q[5],q[7];
ry(-2.9942141370441373) q[5];
ry(0.8779599641778715) q[7];
cx q[5],q[7];
ry(0.043818393954409444) q[7];
ry(-1.2541118000598228) q[9];
cx q[7],q[9];
ry(-0.9022326124626225) q[7];
ry(2.0664439768290936) q[9];
cx q[7],q[9];
ry(-0.043627418862253305) q[9];
ry(-3.102665675579364) q[11];
cx q[9],q[11];
ry(-3.084307504927824) q[9];
ry(0.5368997233661033) q[11];
cx q[9],q[11];
ry(-0.3226983973234685) q[0];
ry(1.0493294677972589) q[1];
ry(-1.0742485556513852) q[2];
ry(0.2542193523557117) q[3];
ry(2.1566068431114536) q[4];
ry(2.723132276627084) q[5];
ry(-1.2713724417548768) q[6];
ry(-0.6234163637159194) q[7];
ry(0.3861708260799715) q[8];
ry(2.4661945023221663) q[9];
ry(0.800768540318554) q[10];
ry(-0.7096523232208165) q[11];