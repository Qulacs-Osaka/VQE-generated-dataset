OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(-0.04732200873152337) q[0];
rz(-0.8246381475714119) q[0];
ry(-0.0726062494593549) q[1];
rz(2.6960287948710637) q[1];
ry(-3.063210522572372) q[2];
rz(-2.3869642810542993) q[2];
ry(3.1403919511132172) q[3];
rz(-1.6290758578272515) q[3];
ry(-3.1049146423301837) q[4];
rz(0.12554830662591865) q[4];
ry(-1.5198751190721554e-05) q[5];
rz(-2.050497595608894) q[5];
ry(-1.4513601837317605) q[6];
rz(-0.484337498546001) q[6];
ry(-0.06105742571590511) q[7];
rz(0.0017078181267024158) q[7];
ry(0.11045214259020943) q[8];
rz(-2.8635062925167243) q[8];
ry(1.2219091648554155) q[9];
rz(1.4694699411745011) q[9];
ry(-1.3659061526862906) q[10];
rz(-2.370011295919098) q[10];
ry(3.1395764417948437) q[11];
rz(0.22937478525445165) q[11];
ry(0.026734902438023234) q[12];
rz(-2.7004894253039717) q[12];
ry(0.044446137904098196) q[13];
rz(-2.7800930597090585) q[13];
ry(-0.26454769069619305) q[14];
rz(0.6529883219879506) q[14];
ry(1.0784557719769916) q[15];
rz(3.0679542283320247) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.9378237017046095) q[0];
rz(2.451267519414133) q[0];
ry(-3.126572532259593) q[1];
rz(0.5202762986691472) q[1];
ry(1.4372459430632878) q[2];
rz(2.4052574956354014) q[2];
ry(-0.5681893655218078) q[3];
rz(2.313133898021625) q[3];
ry(-2.728369516159609) q[4];
rz(3.1290589789773935) q[4];
ry(1.5707311715805359) q[5];
rz(-1.8252205001072408) q[5];
ry(-0.13502922328490863) q[6];
rz(-2.8765663077356023) q[6];
ry(-3.134337600911279) q[7];
rz(-2.8171585470317515) q[7];
ry(-2.2460136874754237) q[8];
rz(0.032136580367672814) q[8];
ry(-3.0433677605495535) q[9];
rz(-1.4686190319269734) q[9];
ry(-2.1584553294346236) q[10];
rz(3.0439587905945116) q[10];
ry(3.13483803109808) q[11];
rz(2.0093333012101664) q[11];
ry(3.108074651105244) q[12];
rz(2.8285639087590653) q[12];
ry(0.8947524681634764) q[13];
rz(1.5491350443910274) q[13];
ry(-2.398944848670038) q[14];
rz(-1.2429358203293146) q[14];
ry(-2.1221933499286108) q[15];
rz(-2.3284014499292076) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.4098752472994107) q[0];
rz(2.075234197727702) q[0];
ry(-0.5614730650690131) q[1];
rz(-2.2890160196445803) q[1];
ry(-3.1413498452478636) q[2];
rz(-0.5398285891801281) q[2];
ry(-3.1414848722114086) q[3];
rz(2.3109747905371134) q[3];
ry(1.5707477395956848) q[4];
rz(-1.6553056794317893) q[4];
ry(-2.752473618563653) q[5];
rz(-2.594971627081854) q[5];
ry(0.13894057640808652) q[6];
rz(3.048441713917459) q[6];
ry(1.6059924785349544) q[7];
rz(0.7896086426765435) q[7];
ry(-0.504780733477556) q[8];
rz(-2.9221788641782065) q[8];
ry(0.24033687870793405) q[9];
rz(-3.0714052345781675) q[9];
ry(-0.2440439582479445) q[10];
rz(-1.2027473999595515) q[10];
ry(-0.012117359507595897) q[11];
rz(-0.9118633891989399) q[11];
ry(0.017447964918985034) q[12];
rz(-1.8633771890154158) q[12];
ry(2.777270733797957) q[13];
rz(-1.0634377964832282) q[13];
ry(0.40940387058095135) q[14];
rz(3.1172230357670734) q[14];
ry(-1.5645695503746646) q[15];
rz(-1.7647096792316637) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.816552684402244) q[0];
rz(-1.718124292069609) q[0];
ry(-0.8549059278857651) q[1];
rz(1.9701832687936016) q[1];
ry(1.4052586994993927) q[2];
rz(2.43521956427506) q[2];
ry(1.5707491527516253) q[3];
rz(1.7508972403938543) q[3];
ry(2.797158529433172) q[4];
rz(1.2160918174686215) q[4];
ry(0.7127082537580209) q[5];
rz(-0.0998531724226948) q[5];
ry(-0.001365110480651575) q[6];
rz(-2.7163216346825347) q[6];
ry(0.611762395157978) q[7];
rz(3.0994392768212102) q[7];
ry(1.5251065395828414) q[8];
rz(1.9884721366615032) q[8];
ry(0.15775205244416934) q[9];
rz(-0.8349003814782029) q[9];
ry(-0.7314863918904498) q[10];
rz(-1.7772223025126683) q[10];
ry(0.029477275939894118) q[11];
rz(2.366978097879136) q[11];
ry(0.008876201594199884) q[12];
rz(-0.2809972848897534) q[12];
ry(-2.5436222455018886) q[13];
rz(2.733693258864302) q[13];
ry(0.6216738563032083) q[14];
rz(-1.910157129364532) q[14];
ry(-2.8390469036322283) q[15];
rz(-1.491062940970504) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-3.1378532845022042) q[0];
rz(0.534196287354014) q[0];
ry(2.1511001970906305) q[1];
rz(-0.8218460549437276) q[1];
ry(1.5413693743811068) q[2];
rz(-0.1470804843458735) q[2];
ry(3.1330588990181885) q[3];
rz(-0.3838380807166377) q[3];
ry(2.205914599212943) q[4];
rz(-2.417908681069885) q[4];
ry(-0.15929159440857357) q[5];
rz(0.6462619133222952) q[5];
ry(-3.1414007191630806) q[6];
rz(1.8353761577561376) q[6];
ry(3.0508749622045035) q[7];
rz(-0.04571569956827186) q[7];
ry(-0.013130564566002352) q[8];
rz(1.7435948044395833) q[8];
ry(-1.6578245960614084) q[9];
rz(-2.3378451749728124) q[9];
ry(2.009096138474847) q[10];
rz(-1.452285135164574) q[10];
ry(3.1071889111296778) q[11];
rz(-0.5228868673471226) q[11];
ry(-0.5276537485730914) q[12];
rz(0.5649154184770393) q[12];
ry(0.27742337073144085) q[13];
rz(-1.6036116358664767) q[13];
ry(1.5689746415633836) q[14];
rz(1.6950361958456117) q[14];
ry(2.775393442063313) q[15];
rz(-2.129559134252145) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.06292931120597078) q[0];
rz(-3.1387678772209617) q[0];
ry(1.5207442109638531) q[1];
rz(-3.01669243189031) q[1];
ry(-2.6604604090323054) q[2];
rz(2.9033761886608582) q[2];
ry(0.0033711237392699545) q[3];
rz(1.127345927117859) q[3];
ry(0.44482500446713935) q[4];
rz(-1.5312049195886297) q[4];
ry(-3.047808310804535) q[5];
rz(3.0382438428560903) q[5];
ry(0.00042870606813681677) q[6];
rz(0.359053299121051) q[6];
ry(-0.6212697996216727) q[7];
rz(1.6700629988765954) q[7];
ry(-3.1230894918793553) q[8];
rz(-2.827426802210276) q[8];
ry(3.140828307492414) q[9];
rz(2.4597577126921966) q[9];
ry(0.05898463283075905) q[10];
rz(-2.0759116724369444) q[10];
ry(-3.140516121298387) q[11];
rz(3.071180693843656) q[11];
ry(0.027314186094202567) q[12];
rz(2.612674540971488) q[12];
ry(-0.008413527484765204) q[13];
rz(-2.618131521416613) q[13];
ry(-2.597540210589739) q[14];
rz(0.018674756191304095) q[14];
ry(1.2501959116117698) q[15];
rz(1.5449070342701887) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.9962068446355248) q[0];
rz(-1.5484496679573727) q[0];
ry(-2.126093615477685) q[1];
rz(-0.6915779694876649) q[1];
ry(2.6680864255388346) q[2];
rz(-0.09363077878121107) q[2];
ry(3.1414548945796676) q[3];
rz(-0.8285954541868952) q[3];
ry(-0.32924931986929007) q[4];
rz(-2.69666635016029) q[4];
ry(-3.0465936260276636) q[5];
rz(1.9617259238016824) q[5];
ry(-0.01574148310529999) q[6];
rz(-2.1440968003388843) q[6];
ry(-2.022222618908708) q[7];
rz(-0.9888855155579589) q[7];
ry(-2.127980073272511) q[8];
rz(0.5592265005961803) q[8];
ry(2.2344901641606443) q[9];
rz(-2.6755167054705624) q[9];
ry(-0.10430884785544058) q[10];
rz(-0.9768020452615377) q[10];
ry(3.1412542638105387) q[11];
rz(-0.16283828889815677) q[11];
ry(-2.641481566240749) q[12];
rz(-0.4171110175352126) q[12];
ry(1.9377937957031133) q[13];
rz(-1.8631450426730645) q[13];
ry(-2.7792116308391095) q[14];
rz(-1.2256258568184943) q[14];
ry(-2.519793707079428) q[15];
rz(-0.06195843894567767) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(2.7717689561549346) q[0];
rz(3.0807436710115397) q[0];
ry(0.0017335752387070968) q[1];
rz(1.1745678888343845) q[1];
ry(-1.2163647609434303) q[2];
rz(2.595124720090482) q[2];
ry(-3.1211696098816724) q[3];
rz(-2.3359886397663363) q[3];
ry(-1.2800483305597028) q[4];
rz(-1.9553158769281669) q[4];
ry(2.2526743389931685) q[5];
rz(-0.3452166555938562) q[5];
ry(0.010162463149060233) q[6];
rz(2.1879153403152434) q[6];
ry(-0.5346544868599202) q[7];
rz(-2.4354734070068047) q[7];
ry(-1.1144031254569977) q[8];
rz(2.223203368819597) q[8];
ry(-3.1235884954244684) q[9];
rz(0.009762205482347854) q[9];
ry(-1.5335106006629664) q[10];
rz(-1.8270110825459094) q[10];
ry(3.1179875294697728) q[11];
rz(-1.0461133667742193) q[11];
ry(0.017945533315200296) q[12];
rz(-1.5247765277564644) q[12];
ry(-0.4972279982506001) q[13];
rz(-0.9558838710194939) q[13];
ry(1.0490900669808314) q[14];
rz(-0.5099756864129309) q[14];
ry(0.3284995431522581) q[15];
rz(-0.004424187742967785) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.742871213772295) q[0];
rz(0.025357547531007718) q[0];
ry(-1.5229330575039077) q[1];
rz(1.9525016889368114) q[1];
ry(3.0811374249918195) q[2];
rz(-0.08005905011709585) q[2];
ry(0.018367577248200855) q[3];
rz(-2.6201844957159017) q[3];
ry(2.910204023919876) q[4];
rz(-1.8798147606479958) q[4];
ry(0.021800926282246138) q[5];
rz(0.04249948507127065) q[5];
ry(3.1369750159955796) q[6];
rz(-0.8959024167046218) q[6];
ry(0.16820962923989313) q[7];
rz(2.215772526028002) q[7];
ry(-0.6129876165236103) q[8];
rz(2.704395646734541) q[8];
ry(1.897452173584637) q[9];
rz(-1.4029179724113656) q[9];
ry(-0.3582410913753725) q[10];
rz(-2.5090087880556284) q[10];
ry(1.5800189585641053) q[11];
rz(-2.16887155920422) q[11];
ry(2.3491540888761437) q[12];
rz(-3.0147381796065678) q[12];
ry(0.47243869483484957) q[13];
rz(-1.58799044276938) q[13];
ry(0.5790560638092781) q[14];
rz(-0.9991791684197764) q[14];
ry(0.47530176123236245) q[15];
rz(2.6792173015551666) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(0.8695167042246172) q[0];
rz(-0.20340215992029975) q[0];
ry(2.9077550256191285) q[1];
rz(0.8723022798809364) q[1];
ry(-2.980079923518397) q[2];
rz(-2.5128741332590487) q[2];
ry(-3.1332456454844855) q[3];
rz(1.6929454608338137) q[3];
ry(-1.529571659587571) q[4];
rz(2.7014537835054306) q[4];
ry(-2.719549833044005) q[5];
rz(-1.0187076031402615) q[5];
ry(0.00888735103876186) q[6];
rz(-1.1928082239347297) q[6];
ry(-3.091800198439527) q[7];
rz(-0.9162208512476019) q[7];
ry(-0.18003339182985378) q[8];
rz(3.0903285691540403) q[8];
ry(-2.9338520340670935) q[9];
rz(1.7506572581843811) q[9];
ry(-0.2747464612919721) q[10];
rz(-2.336981294836254) q[10];
ry(3.1406691707096117) q[11];
rz(-2.095245423728607) q[11];
ry(3.1330827376590196) q[12];
rz(-3.083224331295905) q[12];
ry(0.06791396588772687) q[13];
rz(1.2966380610398338) q[13];
ry(-2.021152681633767) q[14];
rz(-0.8310330717841593) q[14];
ry(-1.6608324595754649) q[15];
rz(-1.1637180677402874) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(1.7502388896626009) q[0];
rz(-2.082705031006312) q[0];
ry(1.0107096942806395) q[1];
rz(1.7805532740167938) q[1];
ry(-0.0014250172139131931) q[2];
rz(-0.4767535321242127) q[2];
ry(3.1272014863108457) q[3];
rz(-1.8174095186854515) q[3];
ry(1.82602265403737) q[4];
rz(0.521977228529365) q[4];
ry(0.09881921424723165) q[5];
rz(-1.1013928708170129) q[5];
ry(3.1352203437419415) q[6];
rz(1.6116771285912912) q[6];
ry(1.6813777862676762) q[7];
rz(-1.5025650473154766) q[7];
ry(-0.15444562184115185) q[8];
rz(0.2055772845422954) q[8];
ry(2.983772157342943) q[9];
rz(0.246662547665272) q[9];
ry(-3.113192127691505) q[10];
rz(0.5888309222627494) q[10];
ry(-3.131768929716846) q[11];
rz(-2.6891163111831267) q[11];
ry(2.3047834240215783) q[12];
rz(-2.85301512521876) q[12];
ry(0.18487031059348946) q[13];
rz(1.9787327952900715) q[13];
ry(2.502095012149648) q[14];
rz(-3.1000139603043424) q[14];
ry(0.10688654417743002) q[15];
rz(-0.3666828467978796) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.592029830462547) q[0];
rz(-2.6861492154290625) q[0];
ry(-2.588935241845722) q[1];
rz(2.8282755903400694) q[1];
ry(1.038177087721932) q[2];
rz(-2.903325387325921) q[2];
ry(0.06114431549030975) q[3];
rz(1.0599277284629876) q[3];
ry(2.8973389848166886) q[4];
rz(2.4200391105027372) q[4];
ry(-2.5854233261726574) q[5];
rz(-0.2627261158213523) q[5];
ry(-1.821214002788599) q[6];
rz(-1.8525511688169871) q[6];
ry(-2.3646629721723107) q[7];
rz(2.828312829438207) q[7];
ry(1.3736236804878565) q[8];
rz(2.738181844196466) q[8];
ry(-2.154652668464312) q[9];
rz(0.3811373892751125) q[9];
ry(-2.976110816490006) q[10];
rz(2.259667103640232) q[10];
ry(1.620336715972718) q[11];
rz(1.2941740436338813) q[11];
ry(1.5626919418294678) q[12];
rz(-1.5007608207073697) q[12];
ry(2.244335193534088) q[13];
rz(2.075024457488007) q[13];
ry(-1.452517114691793) q[14];
rz(-2.433158888221602) q[14];
ry(2.6728183409494326) q[15];
rz(-2.2290097411521623) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-3.0262175593724776) q[0];
rz(0.909933043412842) q[0];
ry(-3.09202858255341) q[1];
rz(-2.571002952571539) q[1];
ry(-0.00168428447251312) q[2];
rz(-0.7018503719327258) q[2];
ry(3.105866901392699) q[3];
rz(1.6228917675403602) q[3];
ry(0.2862736928191909) q[4];
rz(-1.2452372494907344) q[4];
ry(2.8770113592382183) q[5];
rz(1.6971534734561278) q[5];
ry(0.16909652303039) q[6];
rz(2.921341686244013) q[6];
ry(-0.0042806275364384305) q[7];
rz(-0.7229460308058933) q[7];
ry(1.5946189400763455) q[8];
rz(0.027726697812322953) q[8];
ry(0.05444429296867703) q[9];
rz(-3.0853074038108605) q[9];
ry(3.1140191920698728) q[10];
rz(1.7698736059284617) q[10];
ry(3.101353615755841) q[11];
rz(0.5009037914817683) q[11];
ry(-0.047986218837776456) q[12];
rz(2.1588288284963313) q[12];
ry(-1.5709319445974743) q[13];
rz(-0.27211364568220214) q[13];
ry(1.9592599342004848) q[14];
rz(-1.9635068251068732) q[14];
ry(0.419896404356976) q[15];
rz(1.1427671365940695) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-1.148850616895868) q[0];
rz(1.967103114380535) q[0];
ry(-2.769751308097546) q[1];
rz(0.4352550304583245) q[1];
ry(-1.7557107025048209) q[2];
rz(-2.131641144708577) q[2];
ry(-0.04174109126034653) q[3];
rz(-2.472601837128902) q[3];
ry(-0.18537051057633033) q[4];
rz(-3.091188982332868) q[4];
ry(-3.1104437488499608) q[5];
rz(1.6722932243188702) q[5];
ry(0.18222155264226342) q[6];
rz(0.22688005869258543) q[6];
ry(-3.1401675814655685) q[7];
rz(-0.022581585804663185) q[7];
ry(1.0609777733439971) q[8];
rz(-0.021975370698141813) q[8];
ry(-3.001217637755608) q[9];
rz(1.9658825221018628) q[9];
ry(0.9901557245905765) q[10];
rz(-1.999979859883203) q[10];
ry(1.4255756322130244) q[11];
rz(-1.6783193834227803) q[11];
ry(1.625061277583896) q[12];
rz(-2.533848736562651) q[12];
ry(-1.5368404650713716) q[13];
rz(-2.07959811062258) q[13];
ry(1.5702229253261786) q[14];
rz(0.3524491927711989) q[14];
ry(-0.7651394722905147) q[15];
rz(2.896514769418287) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(2.5686976231109107) q[0];
rz(1.1635433629274257) q[0];
ry(-2.9478770225615447) q[1];
rz(-2.3980611447737368) q[1];
ry(-0.011634830013249164) q[2];
rz(1.9161654973178237) q[2];
ry(-3.091180256143171) q[3];
rz(2.917012774633119) q[3];
ry(-2.6453278568933576) q[4];
rz(-3.098230318930092) q[4];
ry(-2.8605655848871843) q[5];
rz(2.8005515567304093) q[5];
ry(0.17574679462095322) q[6];
rz(-0.11951902261866544) q[6];
ry(-0.007303123521207765) q[7];
rz(2.254836442884797) q[7];
ry(-1.575198992907636) q[8];
rz(1.7861955989427614) q[8];
ry(-0.03932758578508118) q[9];
rz(2.574594678461109) q[9];
ry(3.1283026491676877) q[10];
rz(-0.7480830670581451) q[10];
ry(0.009479592084657436) q[11];
rz(3.0065713677191184) q[11];
ry(0.015291497812148336) q[12];
rz(2.781426160447098) q[12];
ry(-0.00611723677113952) q[13];
rz(2.1718936547519414) q[13];
ry(0.05461082856963136) q[14];
rz(-0.05848667787541859) q[14];
ry(1.5709144310122887) q[15];
rz(-2.89081991971687) q[15];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[12],q[13];
cz q[14],q[15];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
cz q[11],q[12];
cz q[13],q[14];
ry(-0.2344076488736269) q[0];
rz(-1.6092301987185433) q[0];
ry(-1.5297013460002151) q[1];
rz(1.2734049857222605) q[1];
ry(-1.6600656330307118) q[2];
rz(-0.8067766665454253) q[2];
ry(1.544466895249325) q[3];
rz(2.5196931097649053) q[3];
ry(3.0564753371196836) q[4];
rz(0.9627367884022258) q[4];
ry(-0.05035114284628772) q[5];
rz(1.1160572623822833) q[5];
ry(0.47641399261610246) q[6];
rz(-1.8595833742584567) q[6];
ry(-0.4949963630628851) q[7];
rz(2.0840842317876573) q[7];
ry(1.655751927374393) q[8];
rz(0.43577316805887895) q[8];
ry(2.1023870019887267) q[9];
rz(-2.6825431476861885) q[9];
ry(-2.711788704192075) q[10];
rz(-1.437931183750071) q[10];
ry(1.3164035148220803) q[11];
rz(-0.8008970078585295) q[11];
ry(2.145781178332677) q[12];
rz(-0.10171339737483719) q[12];
ry(1.048831085629381) q[13];
rz(2.590566584227973) q[13];
ry(0.45667687732516793) q[14];
rz(2.3306889639418267) q[14];
ry(-2.598690886763242) q[15];
rz(-0.32872936950511666) q[15];