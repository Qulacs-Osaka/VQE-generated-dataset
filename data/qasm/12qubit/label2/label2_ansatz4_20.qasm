OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(-0.451035222247971) q[0];
rz(0.41439882114926263) q[0];
ry(1.278549218306282) q[1];
rz(2.5332217465938567) q[1];
ry(-3.140668867639474) q[2];
rz(0.33920872379869316) q[2];
ry(0.0003553861766641343) q[3];
rz(0.5720984514657577) q[3];
ry(-1.5725910095609874) q[4];
rz(2.9722698753668046) q[4];
ry(1.5641557442825933) q[5];
rz(0.34710102365157536) q[5];
ry(0.0008125313023219505) q[6];
rz(1.4646009715058748) q[6];
ry(-3.140827973303051) q[7];
rz(-0.7253243154481623) q[7];
ry(0.0017833838705296756) q[8];
rz(2.9559424939809054) q[8];
ry(0.0018776162965518858) q[9];
rz(-0.749992233076421) q[9];
ry(-1.5739287798666743) q[10];
rz(-0.031010823138882596) q[10];
ry(-1.5689671797717593) q[11];
rz(-2.655625509159345) q[11];
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
ry(-0.5111078329327023) q[0];
rz(0.1736599616144998) q[0];
ry(0.44833396103737183) q[1];
rz(-0.1360474156624996) q[1];
ry(-3.12910587400484) q[2];
rz(2.588878275782047) q[2];
ry(3.090285178227716) q[3];
rz(-1.3236591726292986) q[3];
ry(2.7380252418052744) q[4];
rz(-2.312885455193758) q[4];
ry(-0.4500944730595444) q[5];
rz(1.1168127444381484) q[5];
ry(-3.1403022937046954) q[6];
rz(-1.8139059154919095) q[6];
ry(-0.019520291763287286) q[7];
rz(2.1372598549771507) q[7];
ry(-1.2071782275122045) q[8];
rz(-2.3456089236395905) q[8];
ry(0.4021347578891685) q[9];
rz(2.3783181670803786) q[9];
ry(2.952733582874187) q[10];
rz(3.002999571544422) q[10];
ry(1.2544991400532242) q[11];
rz(0.4708251309367881) q[11];
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
ry(1.9847583005164458) q[0];
rz(2.009276303192511) q[0];
ry(0.02253315318511362) q[1];
rz(2.2548124754938073) q[1];
ry(3.1406189208712076) q[2];
rz(1.9544549901664707) q[2];
ry(-0.0007086709134356455) q[3];
rz(3.0800292125362065) q[3];
ry(-3.0876122912874755) q[4];
rz(1.3985782649285499) q[4];
ry(-0.00867399682503045) q[5];
rz(-2.547866696408038) q[5];
ry(0.0031725746286666023) q[6];
rz(-1.0522103125078015) q[6];
ry(3.141111458812778) q[7];
rz(-2.245940687902874) q[7];
ry(-0.014212954955843383) q[8];
rz(2.9454740721942296) q[8];
ry(-3.1240255684616125) q[9];
rz(2.435118653895257) q[9];
ry(-3.139625238118386) q[10];
rz(-1.3865944426542356) q[10];
ry(0.0046699442816777025) q[11];
rz(-1.807750753716011) q[11];
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
ry(-1.4632554386970738) q[0];
rz(0.18416120045067785) q[0];
ry(1.8413113802106307) q[1];
rz(-1.7053353668461098) q[1];
ry(1.7933411013658747) q[2];
rz(1.8468376921993181) q[2];
ry(1.5095329996831033) q[3];
rz(2.2657193862871576) q[3];
ry(-2.2688465683543373) q[4];
rz(1.700771414451726) q[4];
ry(3.0161214543945536) q[5];
rz(0.4129142261735374) q[5];
ry(3.1104612718431652) q[6];
rz(-1.9994746182893435) q[6];
ry(3.125352092718815) q[7];
rz(0.18213449406684032) q[7];
ry(-2.75610120094667) q[8];
rz(2.319067937539795) q[8];
ry(-2.0205031083728993) q[9];
rz(-0.5476673500550086) q[9];
ry(2.890025701537073) q[10];
rz(1.571482714419389) q[10];
ry(-2.891148444860462) q[11];
rz(-1.5071666511864876) q[11];
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
ry(-2.0773996548674694) q[0];
rz(-1.7685608334188192) q[0];
ry(-2.692845885022474) q[1];
rz(-2.1108006521362146) q[1];
ry(-0.054073605577585226) q[2];
rz(-0.323163520364047) q[2];
ry(0.20318883459358617) q[3];
rz(-2.826910639333778) q[3];
ry(3.0200106156288715) q[4];
rz(-1.613631997032361) q[4];
ry(-3.0235459042741843) q[5];
rz(1.2924955604246315) q[5];
ry(-0.4391090943030136) q[6];
rz(2.085062562522678) q[6];
ry(-2.9752836263329487) q[7];
rz(-2.5563368345712387) q[7];
ry(-2.5837568051859208) q[8];
rz(-3.0161264842298876) q[8];
ry(0.00848950893335676) q[9];
rz(2.682664717390353) q[9];
ry(1.0999760671087409) q[10];
rz(1.3943158532265363) q[10];
ry(2.0383924498171226) q[11];
rz(1.734596535198347) q[11];
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
ry(0.012628425960741652) q[0];
rz(3.100269014954739) q[0];
ry(2.6994477710244777) q[1];
rz(-1.6711162259661867) q[1];
ry(0.01578284020382851) q[2];
rz(-1.2623183215061813) q[2];
ry(0.00965375796497625) q[3];
rz(0.585329363452089) q[3];
ry(2.993139907324237) q[4];
rz(-2.217682741024493) q[4];
ry(2.9914780041413924) q[5];
rz(-2.3751683055430513) q[5];
ry(0.04413472401321095) q[6];
rz(-0.6866883777896092) q[6];
ry(-3.1314141039333117) q[7];
rz(0.9950837502113743) q[7];
ry(1.660487046813564) q[8];
rz(1.6041413602907728) q[8];
ry(1.1632140295556965) q[9];
rz(-1.7586603470339615) q[9];
ry(2.7106229574624816) q[10];
rz(1.2271434901633587) q[10];
ry(0.45657931597996176) q[11];
rz(1.1637058234904494) q[11];
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
ry(-1.546712545900708) q[0];
rz(-2.672157043800305) q[0];
ry(2.798004952745216) q[1];
rz(-2.913635220510286) q[1];
ry(1.104566502154734) q[2];
rz(0.0895942927572913) q[2];
ry(1.5447522097977882) q[3];
rz(-2.4638233485200747) q[3];
ry(1.1359872698979967) q[4];
rz(0.3224083925123322) q[4];
ry(1.6552942188028328) q[5];
rz(3.1322126141378677) q[5];
ry(1.7576718003214689) q[6];
rz(-3.128930701343513) q[6];
ry(-1.3527317622349555) q[7];
rz(0.4586011526534208) q[7];
ry(0.0312983354530072) q[8];
rz(1.6319831091616217) q[8];
ry(-0.3283591877064088) q[9];
rz(-1.4555494890150937) q[9];
ry(2.8927287890321796) q[10];
rz(0.9055077702962357) q[10];
ry(0.2138231038948195) q[11];
rz(-2.860659128298678) q[11];
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
ry(0.6212322762797307) q[0];
rz(2.244395930438243) q[0];
ry(-1.1228321856587478) q[1];
rz(1.7059470006212258) q[1];
ry(0.0010486724467733462) q[2];
rz(-0.325870467231909) q[2];
ry(0.0005903687857644613) q[3];
rz(2.9912668136231) q[3];
ry(-3.1108871154103386) q[4];
rz(-1.479268472505157) q[4];
ry(-3.1109719579986588) q[5];
rz(-2.144205265388445) q[5];
ry(3.14125814491039) q[6];
rz(1.5481370375972918) q[6];
ry(3.1415021818407363) q[7];
rz(2.804704619401214) q[7];
ry(-1.223030156962091) q[8];
rz(1.2220280127954208) q[8];
ry(-1.221012722847832) q[9];
rz(-1.2130733312146336) q[9];
ry(-0.05353543067294364) q[10];
rz(-1.405334379202558) q[10];
ry(0.02611698823192924) q[11];
rz(-1.5935667142917187) q[11];
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
ry(-2.990607841128403) q[0];
rz(-1.2088007075589795) q[0];
ry(1.6544889565916163) q[1];
rz(-1.2039804207359444) q[1];
ry(1.7121021707833834) q[2];
rz(1.3283956271466855) q[2];
ry(-1.1955972794827048) q[3];
rz(-2.125911639110991) q[3];
ry(-0.3745435600295295) q[4];
rz(-0.8092351191622225) q[4];
ry(1.1017475848603155) q[5];
rz(-2.495281923500101) q[5];
ry(-3.0612870204655036) q[6];
rz(0.8085948190954744) q[6];
ry(-0.05315185300658776) q[7];
rz(0.8053335622739857) q[7];
ry(0.6619516281925932) q[8];
rz(1.1110589009427088) q[8];
ry(0.8240063376318796) q[9];
rz(2.9133687325683977) q[9];
ry(-2.0381638733405283) q[10];
rz(2.512836467839641) q[10];
ry(1.02760104688008) q[11];
rz(-1.034085193963472) q[11];
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
ry(0.39065677353164846) q[0];
rz(1.40521219571131) q[0];
ry(0.4374659839462295) q[1];
rz(2.556559383743173) q[1];
ry(1.5376966176605593) q[2];
rz(-1.7022780271337816) q[2];
ry(-2.9885432774885707) q[3];
rz(-2.4450724166267475) q[3];
ry(-1.5569031743673467) q[4];
rz(1.1782554913920045) q[4];
ry(-2.2761936315460902) q[5];
rz(-2.2374253327150995) q[5];
ry(-2.417706527785726) q[6];
rz(1.1735794888648199) q[6];
ry(-1.9379978996144143) q[7];
rz(0.635223665900618) q[7];
ry(-3.1393662918125793) q[8];
rz(-1.6238747029166931) q[8];
ry(0.0005754944055439094) q[9];
rz(2.7573374111300226) q[9];
ry(0.8043427263474429) q[10];
rz(-2.4334394395754546) q[10];
ry(-3.1357329272887604) q[11];
rz(2.0291483821192564) q[11];
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
ry(-2.2421751538106314) q[0];
rz(3.1148651605076823) q[0];
ry(-0.8412052657523129) q[1];
rz(-2.2494663323801305) q[1];
ry(-0.008286542153887133) q[2];
rz(-3.0820599882595423) q[2];
ry(3.107421604708034) q[3];
rz(1.2614282456981423) q[3];
ry(-0.00800166174301905) q[4];
rz(0.4668457352404769) q[4];
ry(0.013549299090184905) q[5];
rz(1.3108903111006311) q[5];
ry(3.096045430411591) q[6];
rz(-1.7152799495427926) q[6];
ry(3.141477423800086) q[7];
rz(0.6035571218788075) q[7];
ry(-2.981917122223187) q[8];
rz(-1.8128098641903003) q[8];
ry(-0.20969347288309415) q[9];
rz(-0.8400901829351141) q[9];
ry(1.7605377018221637) q[10];
rz(2.863696187285823) q[10];
ry(2.2454009264595762) q[11];
rz(-0.26414873126461474) q[11];
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
ry(-3.1282458800843935) q[0];
rz(-1.6246991598611507) q[0];
ry(1.9687678920032519) q[1];
rz(-1.8134369430612458) q[1];
ry(3.1033448770918475) q[2];
rz(3.028112152938114) q[2];
ry(3.089729693415032) q[3];
rz(1.731183529101178) q[3];
ry(-0.18902834844423455) q[4];
rz(0.3458391865289716) q[4];
ry(-0.9873649461675302) q[5];
rz(1.8035845941270523) q[5];
ry(-1.7401615702045035) q[6];
rz(1.150741871980468) q[6];
ry(-0.3486447020230569) q[7];
rz(1.615525099425855) q[7];
ry(0.04315067317813437) q[8];
rz(-1.5214899214169633) q[8];
ry(-0.03435732520032079) q[9];
rz(-1.9543320818545247) q[9];
ry(0.037645533837873835) q[10];
rz(0.6606128482608146) q[10];
ry(-2.331275026593715) q[11];
rz(1.7608017639644598) q[11];
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
ry(-2.880412026317494) q[0];
rz(-2.8586935334850216) q[0];
ry(-1.6181554856758518) q[1];
rz(-1.5171646243555665) q[1];
ry(2.189186920836262) q[2];
rz(1.837611936422709) q[2];
ry(2.2463683698853574) q[3];
rz(1.687566542136799) q[3];
ry(1.6039303776298937) q[4];
rz(-1.5960332916790811) q[4];
ry(1.6276474776570964) q[5];
rz(-1.5110465379737819) q[5];
ry(-0.05465263937918938) q[6];
rz(1.6219203627826597) q[6];
ry(1.6886092644412467) q[7];
rz(1.5545143708389046) q[7];
ry(-0.06863561982028266) q[8];
rz(-0.9867809356478145) q[8];
ry(0.054724644191062835) q[9];
rz(-1.1029273695151867) q[9];
ry(1.243809525409727) q[10];
rz(-2.5508309408206236) q[10];
ry(1.840799426182903) q[11];
rz(-2.1667589475902416) q[11];
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
ry(0.19699525834219772) q[0];
rz(-1.578063322362976) q[0];
ry(-0.15339503226504192) q[1];
rz(2.9085391599609265) q[1];
ry(-2.030646866511882) q[2];
rz(-0.4009540946765686) q[2];
ry(1.262811699679575) q[3];
rz(2.7955072090414417) q[3];
ry(1.614890456104547) q[4];
rz(-0.8762339734134024) q[4];
ry(1.601730519738644) q[5];
rz(0.9920348704902509) q[5];
ry(-0.6722111797851922) q[6];
rz(-1.2382517665034916) q[6];
ry(-2.635208202075833) q[7];
rz(-1.7432343951439755) q[7];
ry(0.006585465103342791) q[8];
rz(-1.482472807492806) q[8];
ry(-6.680408300052676e-05) q[9];
rz(0.912218231102143) q[9];
ry(-2.666152389407158) q[10];
rz(1.9343925708424852) q[10];
ry(-2.4332327494095547) q[11];
rz(-2.4120891782539076) q[11];
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
ry(-3.073828233282585) q[0];
rz(-1.3824235030876553) q[0];
ry(0.014138242168250367) q[1];
rz(-2.788697726889371) q[1];
ry(-3.038527444159506) q[2];
rz(-2.154533737878327) q[2];
ry(-2.9902616982851122) q[3];
rz(-0.3244795836925398) q[3];
ry(-2.591025780036004) q[4];
rz(-2.7084212523379456) q[4];
ry(2.516151869556116) q[5];
rz(-1.84584916430555) q[5];
ry(-2.211564306030838) q[6];
rz(1.9093705742681857) q[6];
ry(-2.500759485975214) q[7];
rz(-1.3402090206898905) q[7];
ry(-1.786502256739288) q[8];
rz(-1.9014533598227146) q[8];
ry(-1.8859711734950853) q[9];
rz(-1.4461824849220895) q[9];
ry(-3.049560687495538) q[10];
rz(-1.8871398551947205) q[10];
ry(0.9139727645464522) q[11];
rz(-3.1298618261033258) q[11];
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
ry(3.081146239093665) q[0];
rz(-1.5621697445668943) q[0];
ry(3.130692415680616) q[1];
rz(1.381606721902716) q[1];
ry(-2.82523680686589) q[2];
rz(0.4240256006012365) q[2];
ry(-2.6325196844244614) q[3];
rz(2.2443189402515067) q[3];
ry(2.6693667583087137) q[4];
rz(-1.695553517204937) q[4];
ry(-0.036334738253024836) q[5];
rz(-1.9223790026167373) q[5];
ry(-0.3932302391992396) q[6];
rz(1.3486270615772706) q[6];
ry(-0.35840634524446274) q[7];
rz(-1.5195332800052033) q[7];
ry(-3.1193399217328914) q[8];
rz(0.20253202548970975) q[8];
ry(0.011927983406448739) q[9];
rz(-2.1146225839265984) q[9];
ry(3.1364616322360344) q[10];
rz(-0.6569736043897378) q[10];
ry(0.6620582489010154) q[11];
rz(-2.1592952298239605) q[11];
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
ry(-1.0213208114077874) q[0];
rz(-1.5263334405225235) q[0];
ry(2.64362096736042) q[1];
rz(-1.4308920933576) q[1];
ry(-0.06690259021867817) q[2];
rz(-0.7119174810389179) q[2];
ry(-3.088946083328716) q[3];
rz(0.28684177628039104) q[3];
ry(3.050853237908132) q[4];
rz(0.10430769506743279) q[4];
ry(2.1362504232836064) q[5];
rz(-1.505140168539877) q[5];
ry(-3.128094232856314) q[6];
rz(-2.5778488652513945) q[6];
ry(-0.02328428907006774) q[7];
rz(2.343569026279401) q[7];
ry(0.01889879736349639) q[8];
rz(0.8483039657074903) q[8];
ry(-0.09325367645035809) q[9];
rz(2.069391274787579) q[9];
ry(1.4563837785384735) q[10];
rz(-1.397343152202744) q[10];
ry(-1.2143869048832787) q[11];
rz(-2.0070585433352535) q[11];
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
ry(2.353093524293117) q[0];
rz(0.6425864938305992) q[0];
ry(-2.64015308559665) q[1];
rz(-2.6635202483343865) q[1];
ry(3.1415240125102297) q[2];
rz(0.33048530201858295) q[2];
ry(3.1406335347879097) q[3];
rz(2.7478139061860447) q[3];
ry(-1.1871913988475138) q[4];
rz(1.7894016248600095) q[4];
ry(-1.918970616479206) q[5];
rz(-1.0655202099896635) q[5];
ry(-0.05373531911733576) q[6];
rz(-1.7154853085213293) q[6];
ry(-0.054566446336718144) q[7];
rz(2.72890302930805) q[7];
ry(-0.003733708259185533) q[8];
rz(0.5434171619376578) q[8];
ry(-3.1364371687646244) q[9];
rz(-1.7840060716900568) q[9];
ry(2.5962466004286266) q[10];
rz(-2.6905906251945506) q[10];
ry(1.303483270811662) q[11];
rz(-2.493637411688315) q[11];
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
ry(-1.865881828970449) q[0];
rz(2.668192157494445) q[0];
ry(-0.9017484466901955) q[1];
rz(-2.3731122976448744) q[1];
ry(-1.6883105798297962) q[2];
rz(0.8165008882333072) q[2];
ry(-1.7230625811239841) q[3];
rz(-1.9774684827124986) q[3];
ry(0.6959894165347837) q[4];
rz(2.598784596900043) q[4];
ry(-0.09521642420029508) q[5];
rz(0.8875078337384926) q[5];
ry(-1.564160944451007) q[6];
rz(-1.6586889817139436) q[6];
ry(-1.4453158966177455) q[7];
rz(3.1232574859892024) q[7];
ry(0.42021252245844387) q[8];
rz(0.2003109985418486) q[8];
ry(-0.8358680189602765) q[9];
rz(-0.30926608418473983) q[9];
ry(0.2698124447851704) q[10];
rz(-1.2000308689492236) q[10];
ry(-0.8281132533063601) q[11];
rz(-2.0949127836930455) q[11];
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
ry(-0.3565860995519845) q[0];
rz(1.466331906754081) q[0];
ry(-0.8043306572452158) q[1];
rz(1.5656806433101467) q[1];
ry(3.119634003327081) q[2];
rz(-0.14815731035658783) q[2];
ry(-3.1254964977333106) q[3];
rz(-0.22085493014877233) q[3];
ry(0.04855176224266877) q[4];
rz(-2.538044996108878) q[4];
ry(-3.1291627843484777) q[5];
rz(2.942088990529631) q[5];
ry(3.1413645286801612) q[6];
rz(0.7906375704280065) q[6];
ry(-3.140778870738634) q[7];
rz(0.9680653288167081) q[7];
ry(-3.140585730289252) q[8];
rz(-2.9020627100063883) q[8];
ry(-3.1406017834621274) q[9];
rz(2.8250237138754515) q[9];
ry(-0.0931933550057639) q[10];
rz(0.691381939976108) q[10];
ry(-3.007037292527758) q[11];
rz(-1.345656502399664) q[11];
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
ry(-2.3162538431061632) q[0];
rz(-1.1087079813930436) q[0];
ry(0.7959453213264593) q[1];
rz(2.3920928862464352) q[1];
ry(-3.1005140672357387) q[2];
rz(-1.1174720852885238) q[2];
ry(3.098627753923155) q[3];
rz(1.380055122657807) q[3];
ry(1.7364077391084098) q[4];
rz(-1.1828832174652286) q[4];
ry(2.867902986717216) q[5];
rz(2.405504050834108) q[5];
ry(0.8373010920135746) q[6];
rz(-1.1385709912922561) q[6];
ry(1.2608973756146524) q[7];
rz(-3.0989478578851424) q[7];
ry(0.4403823769689231) q[8];
rz(2.7414388908180842) q[8];
ry(-0.9040846616120717) q[9];
rz(-2.4234746402732936) q[9];
ry(-1.3930591264971035) q[10];
rz(2.4963689879388116) q[10];
ry(-0.7391522039794429) q[11];
rz(-0.48262978382020394) q[11];
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
ry(0.3020177515206117) q[0];
rz(-0.904170613600631) q[0];
ry(0.21966462646152873) q[1];
rz(-0.3895191123816888) q[1];
ry(-2.523665951239548) q[2];
rz(2.0686726097102124) q[2];
ry(-2.0212205550555677) q[3];
rz(-1.3951901880934026) q[3];
ry(3.103309874317396) q[4];
rz(2.525637298575893) q[4];
ry(0.023054744895313752) q[5];
rz(2.9768985494694506) q[5];
ry(1.6001009597708207) q[6];
rz(-3.0439669537532774) q[6];
ry(1.6191157151812414) q[7];
rz(0.023225086927901373) q[7];
ry(0.04032665519524385) q[8];
rz(1.4704478732462574) q[8];
ry(3.1395104422994096) q[9];
rz(-2.3936799801633417) q[9];
ry(0.7455690619528532) q[10];
rz(-1.1644556863323432) q[10];
ry(-2.0336428413802756) q[11];
rz(-0.10206733450144423) q[11];
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
ry(-1.32215770641462) q[0];
rz(0.6921682850243542) q[0];
ry(-1.310797202327005) q[1];
rz(1.078645444165309) q[1];
ry(-3.014129732904609) q[2];
rz(0.3529450624640156) q[2];
ry(-0.1887180042744676) q[3];
rz(1.552176988722729) q[3];
ry(-3.0565499774502967) q[4];
rz(1.7139952518209662) q[4];
ry(0.08893157578594908) q[5];
rz(-1.4414023953640738) q[5];
ry(-1.5272894626242222) q[6];
rz(1.5639255290020166) q[6];
ry(1.4615491352677155) q[7];
rz(-3.0920332555044645) q[7];
ry(-3.126279770771135) q[8];
rz(0.4840169133386958) q[8];
ry(3.130488228891514) q[9];
rz(-3.08942767888363) q[9];
ry(-0.12875696172639017) q[10];
rz(-1.4176173846580564) q[10];
ry(-3.0033875970326878) q[11];
rz(-1.6169701997719441) q[11];
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
ry(-0.0021529642829988555) q[0];
rz(0.9627967272568779) q[0];
ry(-3.0675666209718266) q[1];
rz(-0.43115439966948715) q[1];
ry(3.0750864968529124) q[2];
rz(0.2104157761750052) q[2];
ry(-0.4122169579093986) q[3];
rz(-1.5466264737240936) q[3];
ry(1.775809575737724) q[4];
rz(1.7169782243837348) q[4];
ry(1.3254949346062999) q[5];
rz(-1.441179704553341) q[5];
ry(-0.6285705155599789) q[6];
rz(2.6976979561897028) q[6];
ry(2.3141257200628655) q[7];
rz(2.425370675753311) q[7];
ry(1.478381828084684) q[8];
rz(0.8711158624745887) q[8];
ry(1.6653364408684563) q[9];
rz(-2.2898458104907724) q[9];
ry(-0.6308453913448586) q[10];
rz(-1.94009786134187) q[10];
ry(0.7430618336949623) q[11];
rz(1.0958170003537593) q[11];