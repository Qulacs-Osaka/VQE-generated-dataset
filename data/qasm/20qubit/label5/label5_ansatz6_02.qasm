OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
ry(-1.4872869455019684) q[0];
ry(-1.6285080155177116) q[1];
cx q[0],q[1];
ry(-0.5279670165057099) q[0];
ry(1.5669162596860806) q[1];
cx q[0],q[1];
ry(-1.4134813550233334) q[1];
ry(1.445612246263193) q[2];
cx q[1],q[2];
ry(-3.041527634437386) q[1];
ry(2.2522560252812545) q[2];
cx q[1],q[2];
ry(-0.13996721512190222) q[2];
ry(2.44439150511868) q[3];
cx q[2],q[3];
ry(-0.05180884194920925) q[2];
ry(1.857446762713301) q[3];
cx q[2],q[3];
ry(-2.3956193724132344) q[3];
ry(1.973765848672004) q[4];
cx q[3],q[4];
ry(1.6225990737008225) q[3];
ry(1.4091920004417975) q[4];
cx q[3],q[4];
ry(1.4711896598047376) q[4];
ry(1.5424183733852628) q[5];
cx q[4],q[5];
ry(0.7602700745524181) q[4];
ry(-0.11540292779095297) q[5];
cx q[4],q[5];
ry(0.8397141216394219) q[5];
ry(-0.2771589108929593) q[6];
cx q[5],q[6];
ry(1.2735677433536496) q[5];
ry(2.6671938456298903) q[6];
cx q[5],q[6];
ry(2.0155810089745936) q[6];
ry(-2.7820723419920754) q[7];
cx q[6],q[7];
ry(1.473482083677663) q[6];
ry(1.2106276355753307) q[7];
cx q[6],q[7];
ry(-0.8509469032526624) q[7];
ry(-0.5141387364792142) q[8];
cx q[7],q[8];
ry(-0.9544197212681117) q[7];
ry(-0.7621872265493372) q[8];
cx q[7],q[8];
ry(2.703216465170622) q[8];
ry(3.008920359564454) q[9];
cx q[8],q[9];
ry(-1.3289468301115632) q[8];
ry(1.442164513870768) q[9];
cx q[8],q[9];
ry(-1.589995832149878) q[9];
ry(-2.9948711928131737) q[10];
cx q[9],q[10];
ry(-1.7537415910968601) q[9];
ry(1.528361454397311) q[10];
cx q[9],q[10];
ry(-2.8401985084495927) q[10];
ry(2.6720023653414717) q[11];
cx q[10],q[11];
ry(1.9986604162763797) q[10];
ry(-1.616231597331277) q[11];
cx q[10],q[11];
ry(-1.670531123983463) q[11];
ry(2.790313044948253) q[12];
cx q[11],q[12];
ry(-2.4423969768099494) q[11];
ry(-2.413246104453371) q[12];
cx q[11],q[12];
ry(-0.2676793222083971) q[12];
ry(-2.8809613952660498) q[13];
cx q[12],q[13];
ry(-2.3599383521351105) q[12];
ry(-1.9114670395562814) q[13];
cx q[12],q[13];
ry(-2.242500020162738) q[13];
ry(3.089304503785825) q[14];
cx q[13],q[14];
ry(-1.9508838533423334) q[13];
ry(1.095736664243935) q[14];
cx q[13],q[14];
ry(0.8431703093728596) q[14];
ry(2.267405469961928) q[15];
cx q[14],q[15];
ry(0.783672248336492) q[14];
ry(-0.2900507874466287) q[15];
cx q[14],q[15];
ry(2.291155984198969) q[15];
ry(0.4146679055563751) q[16];
cx q[15],q[16];
ry(-1.5907650411164926) q[15];
ry(-1.1536316871463645) q[16];
cx q[15],q[16];
ry(0.9656732699324923) q[16];
ry(0.13623331941611738) q[17];
cx q[16],q[17];
ry(2.2672794023815763) q[16];
ry(0.04415231354452055) q[17];
cx q[16],q[17];
ry(0.15582479654832415) q[17];
ry(-2.299520204242865) q[18];
cx q[17],q[18];
ry(1.2654470915678186) q[17];
ry(-1.6177803083992366) q[18];
cx q[17],q[18];
ry(-2.389643005591711) q[18];
ry(-2.613162394338638) q[19];
cx q[18],q[19];
ry(1.9153595713580973) q[18];
ry(3.1373702279218056) q[19];
cx q[18],q[19];
ry(2.2215038404469776) q[0];
ry(1.6622288163753445) q[1];
cx q[0],q[1];
ry(2.756775404414869) q[0];
ry(0.7249062302068257) q[1];
cx q[0],q[1];
ry(2.0487593130757755) q[1];
ry(-2.247195386150273) q[2];
cx q[1],q[2];
ry(1.9833733015518085) q[1];
ry(-1.823437656346357) q[2];
cx q[1],q[2];
ry(1.7350041232708908) q[2];
ry(1.1969368463922587) q[3];
cx q[2],q[3];
ry(-0.047969081516885836) q[2];
ry(-3.131480681695702) q[3];
cx q[2],q[3];
ry(2.1650152134024396) q[3];
ry(0.0273530942119784) q[4];
cx q[3],q[4];
ry(-3.055531125016701) q[3];
ry(-2.0465721834540282) q[4];
cx q[3],q[4];
ry(-1.3124987583322614) q[4];
ry(-1.1062442590501942) q[5];
cx q[4],q[5];
ry(2.1416706977374993) q[4];
ry(-0.11759338240383489) q[5];
cx q[4],q[5];
ry(2.984352559235409) q[5];
ry(-1.4546266704193567) q[6];
cx q[5],q[6];
ry(0.7819208246804129) q[5];
ry(3.100748569083565) q[6];
cx q[5],q[6];
ry(1.578363526329828) q[6];
ry(1.586048233622214) q[7];
cx q[6],q[7];
ry(-1.6558771153826282) q[6];
ry(0.14454297120713805) q[7];
cx q[6],q[7];
ry(1.5955556944632991) q[7];
ry(-1.525245869171247) q[8];
cx q[7],q[8];
ry(-0.6799193334736189) q[7];
ry(1.6219749160936663) q[8];
cx q[7],q[8];
ry(1.0759270615210461) q[8];
ry(-1.4789928250995068) q[9];
cx q[8],q[9];
ry(-1.5619754900948724) q[8];
ry(3.120143537900363) q[9];
cx q[8],q[9];
ry(-1.721538710178517) q[9];
ry(-1.5651265571663382) q[10];
cx q[9],q[10];
ry(-2.352889937166893) q[9];
ry(-0.5093014947931254) q[10];
cx q[9],q[10];
ry(1.5633158313105782) q[10];
ry(-1.2941279002560817) q[11];
cx q[10],q[11];
ry(-0.08494779668297259) q[10];
ry(1.8057677126343374) q[11];
cx q[10],q[11];
ry(-1.8734516183670755) q[11];
ry(1.61529340847314) q[12];
cx q[11],q[12];
ry(0.9943911187460635) q[11];
ry(-2.389481112852602) q[12];
cx q[11],q[12];
ry(1.7214151135290843) q[12];
ry(-1.5530794382994437) q[13];
cx q[12],q[13];
ry(-0.7685957137135148) q[12];
ry(3.1306199420538663) q[13];
cx q[12],q[13];
ry(-1.5752550644172516) q[13];
ry(1.6397259067986507) q[14];
cx q[13],q[14];
ry(1.082451757889626) q[13];
ry(-2.3448431903794282) q[14];
cx q[13],q[14];
ry(1.48208271512199) q[14];
ry(1.603125144339204) q[15];
cx q[14],q[15];
ry(-1.457177184076434) q[14];
ry(1.519518227539072) q[15];
cx q[14],q[15];
ry(-1.635251550580134) q[15];
ry(1.0888143506378551) q[16];
cx q[15],q[16];
ry(1.5072498528449707) q[15];
ry(1.8062392970080468) q[16];
cx q[15],q[16];
ry(1.5800018669394031) q[16];
ry(-1.6208399397160607) q[17];
cx q[16],q[17];
ry(1.199813675269528) q[16];
ry(-0.6103752059014758) q[17];
cx q[16],q[17];
ry(1.5795674422047663) q[17];
ry(0.5194033132408699) q[18];
cx q[17],q[18];
ry(1.9008463380255012) q[17];
ry(-3.0475008452279586) q[18];
cx q[17],q[18];
ry(-1.4854018080921711) q[18];
ry(-1.2445543995213892) q[19];
cx q[18],q[19];
ry(2.5947178709415866) q[18];
ry(3.134602766192934) q[19];
cx q[18],q[19];
ry(2.007373613442756) q[0];
ry(-2.7103109113753376) q[1];
cx q[0],q[1];
ry(1.0117997417296185) q[0];
ry(1.5682139156848738) q[1];
cx q[0],q[1];
ry(2.74861174319425) q[1];
ry(-1.4866805897456974) q[2];
cx q[1],q[2];
ry(-2.3387602294828063) q[1];
ry(2.111803250312757) q[2];
cx q[1],q[2];
ry(0.6677334635180774) q[2];
ry(1.929020471075539) q[3];
cx q[2],q[3];
ry(1.296751213950162) q[2];
ry(2.709128413201745) q[3];
cx q[2],q[3];
ry(-1.5549487688343022) q[3];
ry(-2.7018964731722606) q[4];
cx q[3],q[4];
ry(1.7660840134645355) q[3];
ry(1.7711158376462215) q[4];
cx q[3],q[4];
ry(1.5730571999256933) q[4];
ry(-0.15422878356623973) q[5];
cx q[4],q[5];
ry(1.5685485447646383) q[4];
ry(1.8142252026334003) q[5];
cx q[4],q[5];
ry(-1.5649705147278379) q[5];
ry(1.4798277442268635) q[6];
cx q[5],q[6];
ry(-1.657003364517605) q[5];
ry(1.155557537745354) q[6];
cx q[5],q[6];
ry(-1.5720532058752745) q[6];
ry(1.5772839545857764) q[7];
cx q[6],q[7];
ry(-1.8163712257905757) q[6];
ry(2.264618736122834) q[7];
cx q[6],q[7];
ry(-1.5579490352528023) q[7];
ry(-2.192455415438763) q[8];
cx q[7],q[8];
ry(0.6373848164394893) q[7];
ry(1.4056344028382886) q[8];
cx q[7],q[8];
ry(1.570195126117703) q[8];
ry(1.5403579132402065) q[9];
cx q[8],q[9];
ry(-1.4814409558982806) q[8];
ry(-1.7860437686726653) q[9];
cx q[8],q[9];
ry(1.6323409969502665) q[9];
ry(1.5655067697614387) q[10];
cx q[9],q[10];
ry(1.4396241681412505) q[9];
ry(-3.0509122871590777) q[10];
cx q[9],q[10];
ry(1.563527164213844) q[10];
ry(1.5616541171431253) q[11];
cx q[10],q[11];
ry(1.8194979561202054) q[10];
ry(-2.6732505251558667) q[11];
cx q[10],q[11];
ry(1.5648074736219488) q[11];
ry(-1.4117100098414215) q[12];
cx q[11],q[12];
ry(-1.662425319492809) q[11];
ry(0.508419331366758) q[12];
cx q[11],q[12];
ry(-2.6417155532332575) q[12];
ry(1.5886749952081232) q[13];
cx q[12],q[13];
ry(1.688506329816755) q[12];
ry(0.001996721584832078) q[13];
cx q[12],q[13];
ry(-1.5747753637715272) q[13];
ry(-1.6158612994412431) q[14];
cx q[13],q[14];
ry(2.368512563412803) q[13];
ry(3.0950953792766405) q[14];
cx q[13],q[14];
ry(-3.0593375142866543) q[14];
ry(1.5874333370534428) q[15];
cx q[14],q[15];
ry(-2.1581595009577974) q[14];
ry(0.00822009660274503) q[15];
cx q[14],q[15];
ry(1.5690726053374613) q[15];
ry(1.055575242189195) q[16];
cx q[15],q[16];
ry(3.1337383897672537) q[15];
ry(-1.4241345393062668) q[16];
cx q[15],q[16];
ry(1.0584209012741628) q[16];
ry(-1.558338733948701) q[17];
cx q[16],q[17];
ry(1.1871170272931961) q[16];
ry(-3.0514411477165155) q[17];
cx q[16],q[17];
ry(1.569188980780273) q[17];
ry(-1.4754253586922363) q[18];
cx q[17],q[18];
ry(-2.3608380845579524) q[17];
ry(-1.7174817615177884) q[18];
cx q[17],q[18];
ry(2.268314754423555) q[18];
ry(-3.0830472833152474) q[19];
cx q[18],q[19];
ry(1.1308635081221647) q[18];
ry(-1.0370131849014532) q[19];
cx q[18],q[19];
ry(-2.196594595284738) q[0];
ry(-2.7057096813682007) q[1];
cx q[0],q[1];
ry(-2.8749920861211042) q[0];
ry(-2.9604753063104545) q[1];
cx q[0],q[1];
ry(1.3200116710597847) q[1];
ry(-0.3596393427002358) q[2];
cx q[1],q[2];
ry(-3.1030417223545865) q[1];
ry(3.1162916981143227) q[2];
cx q[1],q[2];
ry(-0.3134862139180625) q[2];
ry(-1.5678345102913758) q[3];
cx q[2],q[3];
ry(-2.208825148962134) q[2];
ry(-1.114055784283147) q[3];
cx q[2],q[3];
ry(1.5770985366208343) q[3];
ry(1.5767193994436524) q[4];
cx q[3],q[4];
ry(-2.0020164498604753) q[3];
ry(-2.4375903512335957) q[4];
cx q[3],q[4];
ry(1.5590522612871254) q[4];
ry(0.48448631972413975) q[5];
cx q[4],q[5];
ry(1.258013584942965) q[4];
ry(2.3859405667141957) q[5];
cx q[4],q[5];
ry(2.2457311883338624) q[5];
ry(-2.857897984189335) q[6];
cx q[5],q[6];
ry(-0.007399557800936097) q[5];
ry(-0.007760677013831874) q[6];
cx q[5],q[6];
ry(2.877975955264739) q[6];
ry(-1.5394532583848992) q[7];
cx q[6],q[7];
ry(1.2725517056141953) q[6];
ry(2.7344306225096964) q[7];
cx q[6],q[7];
ry(-1.5623089490441888) q[7];
ry(-1.5711783751434134) q[8];
cx q[7],q[8];
ry(-0.8353988963844436) q[7];
ry(-0.027397664781965998) q[8];
cx q[7],q[8];
ry(-1.5632538014309105) q[8];
ry(-1.465132942868836) q[9];
cx q[8],q[9];
ry(-0.5236951706983657) q[8];
ry(0.9431947022870721) q[9];
cx q[8],q[9];
ry(-1.5729115352232235) q[9];
ry(1.5745669643836602) q[10];
cx q[9],q[10];
ry(1.0990094804320503) q[9];
ry(1.5738745699002594) q[10];
cx q[9],q[10];
ry(-1.5666986835200822) q[10];
ry(-1.560711403749428) q[11];
cx q[10],q[11];
ry(-1.6156369899718044) q[10];
ry(-2.3145237541382606) q[11];
cx q[10],q[11];
ry(-1.5785619416605368) q[11];
ry(2.646843666266366) q[12];
cx q[11],q[12];
ry(-1.4621762551308526) q[11];
ry(0.7454676325528682) q[12];
cx q[11],q[12];
ry(-1.5681831753932798) q[12];
ry(1.5754196660366313) q[13];
cx q[12],q[13];
ry(1.0733838637427615) q[12];
ry(-1.8986928278974862) q[13];
cx q[12],q[13];
ry(1.6249795866203296) q[13];
ry(-1.723677021178287) q[14];
cx q[13],q[14];
ry(0.007772824265156686) q[13];
ry(0.36524372351611695) q[14];
cx q[13],q[14];
ry(-3.094739518355281) q[14];
ry(2.627150232667555) q[15];
cx q[14],q[15];
ry(0.003350496426402394) q[14];
ry(-1.6100886100990737) q[15];
cx q[14],q[15];
ry(0.5203337806374124) q[15];
ry(1.5687875117802) q[16];
cx q[15],q[16];
ry(1.271087920747302) q[15];
ry(-1.7202005185769105) q[16];
cx q[15],q[16];
ry(-1.5639253779087197) q[16];
ry(1.5677099760953328) q[17];
cx q[16],q[17];
ry(-1.73033518724) q[16];
ry(-2.5373627138492525) q[17];
cx q[16],q[17];
ry(-1.4463630305714856) q[17];
ry(-2.6608353833493714) q[18];
cx q[17],q[18];
ry(-1.3421588433164038) q[17];
ry(-0.11180240508086836) q[18];
cx q[17],q[18];
ry(-1.5789342609162131) q[18];
ry(-2.74299802073504) q[19];
cx q[18],q[19];
ry(-0.01278592226414123) q[18];
ry(0.7754297156971486) q[19];
cx q[18],q[19];
ry(1.5474928893537987) q[0];
ry(2.965745946444265) q[1];
cx q[0],q[1];
ry(-2.0487651930761657) q[0];
ry(0.2909404040981376) q[1];
cx q[0],q[1];
ry(-2.933259525577072) q[1];
ry(1.5719080175643052) q[2];
cx q[1],q[2];
ry(-0.016719823297237306) q[1];
ry(-0.0824858676767688) q[2];
cx q[1],q[2];
ry(1.55746824338759) q[2];
ry(1.5733486280731719) q[3];
cx q[2],q[3];
ry(-0.6354699834648398) q[2];
ry(-1.8819497286191273) q[3];
cx q[2],q[3];
ry(1.5936436732562669) q[3];
ry(2.596693845979926) q[4];
cx q[3],q[4];
ry(-0.013284926275410667) q[3];
ry(-2.8360985933091283) q[4];
cx q[3],q[4];
ry(0.21334693991367537) q[4];
ry(-1.2530103017014926) q[5];
cx q[4],q[5];
ry(1.2873630444684316) q[4];
ry(-0.030822027292041202) q[5];
cx q[4],q[5];
ry(0.5114166864523613) q[5];
ry(1.5921257345783442) q[6];
cx q[5],q[6];
ry(-1.459303854206655) q[5];
ry(-1.101370881660056) q[6];
cx q[5],q[6];
ry(1.5654044947097507) q[6];
ry(1.5602769376630157) q[7];
cx q[6],q[7];
ry(-1.6576520440875278) q[6];
ry(1.2333520985323503) q[7];
cx q[6],q[7];
ry(1.5654619457457255) q[7];
ry(-1.5727004481514193) q[8];
cx q[7],q[8];
ry(0.7174094629907288) q[7];
ry(-2.345469919427646) q[8];
cx q[7],q[8];
ry(-1.5818232176509648) q[8];
ry(-1.5752658264611676) q[9];
cx q[8],q[9];
ry(-1.6223381759617395) q[8];
ry(-1.6453312308770764) q[9];
cx q[8],q[9];
ry(1.5659681433521069) q[9];
ry(1.563546882748985) q[10];
cx q[9],q[10];
ry(-2.051982927596568) q[9];
ry(2.0787822758057573) q[10];
cx q[9],q[10];
ry(1.567450024872838) q[10];
ry(1.5767360519475169) q[11];
cx q[10],q[11];
ry(-0.5721946150226661) q[10];
ry(1.1473943872590597) q[11];
cx q[10],q[11];
ry(1.5584600919864373) q[11];
ry(1.5855047050544004) q[12];
cx q[11],q[12];
ry(1.3370323458971258) q[11];
ry(2.268611944389229) q[12];
cx q[11],q[12];
ry(1.5700347098830802) q[12];
ry(-3.118030673194851) q[13];
cx q[12],q[13];
ry(-0.002142676954613343) q[12];
ry(-1.0976793743167899) q[13];
cx q[12],q[13];
ry(3.073406259570552) q[13];
ry(-1.5693980039109676) q[14];
cx q[13],q[14];
ry(1.4114906620865648) q[13];
ry(-1.842915156436716) q[14];
cx q[13],q[14];
ry(1.5696602348839914) q[14];
ry(-1.5627518041675956) q[15];
cx q[14],q[15];
ry(-2.51197196880159) q[14];
ry(-0.8435526724852949) q[15];
cx q[14],q[15];
ry(-1.5719575276750926) q[15];
ry(1.5684751049737777) q[16];
cx q[15],q[16];
ry(-1.62537979372606) q[15];
ry(1.4855470150751886) q[16];
cx q[15],q[16];
ry(-1.6040159916823709) q[16];
ry(-1.4765045364118277) q[17];
cx q[16],q[17];
ry(-3.0666541514354493) q[16];
ry(-0.024830771788272088) q[17];
cx q[16],q[17];
ry(1.75246270998593) q[17];
ry(-0.2686171869752069) q[18];
cx q[17],q[18];
ry(-0.9231245726534612) q[17];
ry(-2.724938229882413) q[18];
cx q[17],q[18];
ry(-2.6940208218409714) q[18];
ry(1.8071499404606817) q[19];
cx q[18],q[19];
ry(1.5438655206400511) q[18];
ry(-2.9874129732268107) q[19];
cx q[18],q[19];
ry(3.086352369331169) q[0];
ry(0.9412781521399738) q[1];
ry(1.5831092203394768) q[2];
ry(-1.5537903162807734) q[3];
ry(-1.241915635767965) q[4];
ry(1.5568313293746858) q[5];
ry(-1.5710754861627008) q[6];
ry(-1.5736690669451656) q[7];
ry(1.5718853939318107) q[8];
ry(1.5675772631186207) q[9];
ry(-1.5568908541769204) q[10];
ry(1.5525107080631697) q[11];
ry(1.570226198173641) q[12];
ry(1.5741909550874) q[13];
ry(-1.5697418573530932) q[14];
ry(-1.5683926784219686) q[15];
ry(1.6037274871770026) q[16];
ry(-1.6074252693779671) q[17];
ry(-1.5490149050109328) q[18];
ry(0.8682182900411163) q[19];