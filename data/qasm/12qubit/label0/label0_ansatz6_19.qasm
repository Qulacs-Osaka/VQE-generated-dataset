OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(1.5732857242448621) q[0];
ry(2.312721126997707) q[1];
cx q[0],q[1];
ry(0.3451477779232332) q[0];
ry(-2.6084783863978607) q[1];
cx q[0],q[1];
ry(1.6948476247442323) q[1];
ry(-0.49007447830830037) q[2];
cx q[1],q[2];
ry(-1.0936330059049686) q[1];
ry(-0.15466311633022573) q[2];
cx q[1],q[2];
ry(-2.959124770000889) q[2];
ry(0.6097198124705863) q[3];
cx q[2],q[3];
ry(-0.6836432942258241) q[2];
ry(-0.8498200167950772) q[3];
cx q[2],q[3];
ry(1.446779521423908) q[3];
ry(-2.923506909262465) q[4];
cx q[3],q[4];
ry(1.0388488149293271) q[3];
ry(-2.2484743255590103) q[4];
cx q[3],q[4];
ry(-0.8058315990223373) q[4];
ry(2.040956088455736) q[5];
cx q[4],q[5];
ry(-6.740702377358405e-05) q[4];
ry(-4.223364035560451e-05) q[5];
cx q[4],q[5];
ry(-2.416713098635259) q[5];
ry(1.894084173132756) q[6];
cx q[5],q[6];
ry(-2.866246345281343) q[5];
ry(0.650958882308718) q[6];
cx q[5],q[6];
ry(-1.2625750715156903) q[6];
ry(-0.007447764465465134) q[7];
cx q[6],q[7];
ry(1.6193815338059068) q[6];
ry(1.5457868458313784) q[7];
cx q[6],q[7];
ry(1.6084006024043123) q[7];
ry(0.19061162835418077) q[8];
cx q[7],q[8];
ry(-1.9023166442833828) q[7];
ry(-1.1853333788232856) q[8];
cx q[7],q[8];
ry(-2.206450957624928) q[8];
ry(1.8762855807782017) q[9];
cx q[8],q[9];
ry(0.00016657666035246874) q[8];
ry(-3.1393993225557697) q[9];
cx q[8],q[9];
ry(-1.223965500030885) q[9];
ry(1.4328057670463683) q[10];
cx q[9],q[10];
ry(-2.7994238553632225) q[9];
ry(-0.06653722056113998) q[10];
cx q[9],q[10];
ry(0.4720820740720981) q[10];
ry(-2.9381280068633395) q[11];
cx q[10],q[11];
ry(0.2066156800290324) q[10];
ry(-1.932882736601914) q[11];
cx q[10],q[11];
ry(0.1001015318705809) q[0];
ry(-3.0732585498721168) q[1];
cx q[0],q[1];
ry(0.25235117215235814) q[0];
ry(3.0808705691098104) q[1];
cx q[0],q[1];
ry(-1.028946107522417) q[1];
ry(0.5528505509195701) q[2];
cx q[1],q[2];
ry(-2.641970256818007) q[1];
ry(0.9063777868352655) q[2];
cx q[1],q[2];
ry(-0.25642126795824627) q[2];
ry(2.1692787427746723) q[3];
cx q[2],q[3];
ry(-0.31776411071124644) q[2];
ry(-1.006797852574917) q[3];
cx q[2],q[3];
ry(-3.1356401898765904) q[3];
ry(1.6895173181154979) q[4];
cx q[3],q[4];
ry(-0.3897838175762068) q[3];
ry(0.7121426520571191) q[4];
cx q[3],q[4];
ry(-1.8980282754614581) q[4];
ry(-2.04945894619562) q[5];
cx q[4],q[5];
ry(2.954984340842959) q[4];
ry(-1.0489784064173822) q[5];
cx q[4],q[5];
ry(1.2849744769519156) q[5];
ry(3.0886719265784293) q[6];
cx q[5],q[6];
ry(-2.27686610415692) q[5];
ry(3.1415924318514126) q[6];
cx q[5],q[6];
ry(-0.46510814880634577) q[6];
ry(-3.0145318485463117) q[7];
cx q[6],q[7];
ry(-2.0261156635764603) q[6];
ry(2.593958217099089) q[7];
cx q[6],q[7];
ry(1.3084969194913711) q[7];
ry(-0.8514601024087601) q[8];
cx q[7],q[8];
ry(1.3170075459505037) q[7];
ry(-2.28863597044451) q[8];
cx q[7],q[8];
ry(-1.783771846804071) q[8];
ry(0.37162673827203885) q[9];
cx q[8],q[9];
ry(-2.182487204047251) q[8];
ry(-0.0026964410735108683) q[9];
cx q[8],q[9];
ry(-0.6967756208241961) q[9];
ry(-0.46180585174324146) q[10];
cx q[9],q[10];
ry(-1.8935889578086416) q[9];
ry(-3.027534976879755) q[10];
cx q[9],q[10];
ry(-2.939399676441328) q[10];
ry(-1.5032356295985565) q[11];
cx q[10],q[11];
ry(-0.35674267082738975) q[10];
ry(-1.7715791339844582) q[11];
cx q[10],q[11];
ry(2.24342957591518) q[0];
ry(0.7306691490816667) q[1];
cx q[0],q[1];
ry(-0.017557786577048873) q[0];
ry(-1.0003306556574652) q[1];
cx q[0],q[1];
ry(-0.9668778031235394) q[1];
ry(-0.8658726514569031) q[2];
cx q[1],q[2];
ry(-1.5347055816427109) q[1];
ry(-2.0791325042425672) q[2];
cx q[1],q[2];
ry(0.3531834861125256) q[2];
ry(-2.1811640179145737) q[3];
cx q[2],q[3];
ry(1.5563671798818728) q[2];
ry(2.72118611667044) q[3];
cx q[2],q[3];
ry(2.638845796083691) q[3];
ry(-0.4181543405057067) q[4];
cx q[3],q[4];
ry(-0.2236673978955539) q[3];
ry(-2.95698622768274) q[4];
cx q[3],q[4];
ry(-2.5772515824002302) q[4];
ry(0.9819956564376906) q[5];
cx q[4],q[5];
ry(3.138450689806999) q[4];
ry(2.312731838521729) q[5];
cx q[4],q[5];
ry(2.5119987662852696) q[5];
ry(-3.0185490394193235) q[6];
cx q[5],q[6];
ry(-3.135476743120699) q[5];
ry(1.8554176850942055e-05) q[6];
cx q[5],q[6];
ry(-3.1135394744748766) q[6];
ry(-1.9758008891295782) q[7];
cx q[6],q[7];
ry(-3.0989796348034457) q[6];
ry(-2.4304435951877577) q[7];
cx q[6],q[7];
ry(2.97594274959275) q[7];
ry(1.9807770038767538) q[8];
cx q[7],q[8];
ry(-3.1412756843494334) q[7];
ry(-1.2979189913680058) q[8];
cx q[7],q[8];
ry(0.017712508249781322) q[8];
ry(2.0531281679627615) q[9];
cx q[8],q[9];
ry(0.07027529351914463) q[8];
ry(-0.0035557738531726453) q[9];
cx q[8],q[9];
ry(1.1408684714300676) q[9];
ry(-2.0394319492499475) q[10];
cx q[9],q[10];
ry(-2.0980521171571107) q[9];
ry(-2.5434180711814403) q[10];
cx q[9],q[10];
ry(-0.7774744168134923) q[10];
ry(-2.304584438106186) q[11];
cx q[10],q[11];
ry(3.0166381252304846) q[10];
ry(1.5456132486828822) q[11];
cx q[10],q[11];
ry(0.3831734657915354) q[0];
ry(-1.1529955439295394) q[1];
cx q[0],q[1];
ry(1.5836611278563462) q[0];
ry(1.9602281858097346) q[1];
cx q[0],q[1];
ry(1.6267982495530475) q[1];
ry(-0.6898216040026841) q[2];
cx q[1],q[2];
ry(2.4532331630194344) q[1];
ry(-2.0319864317357923) q[2];
cx q[1],q[2];
ry(1.6443782763399504) q[2];
ry(1.2556988500918866) q[3];
cx q[2],q[3];
ry(1.6062185718872701) q[2];
ry(2.4193883763075474) q[3];
cx q[2],q[3];
ry(-0.45912221287646204) q[3];
ry(-0.4429434687755973) q[4];
cx q[3],q[4];
ry(-1.1044520315953594) q[3];
ry(-2.953380613625136) q[4];
cx q[3],q[4];
ry(-2.0463618769738323) q[4];
ry(-2.4501308940747237) q[5];
cx q[4],q[5];
ry(0.0003302110215051002) q[4];
ry(-3.077288491914584) q[5];
cx q[4],q[5];
ry(0.5602028896920838) q[5];
ry(-2.040802535371942) q[6];
cx q[5],q[6];
ry(0.5692146514345255) q[5];
ry(-3.1415797746102996) q[6];
cx q[5],q[6];
ry(1.4204136131498046) q[6];
ry(-1.666269773058112) q[7];
cx q[6],q[7];
ry(-0.4912665439974946) q[6];
ry(-2.278143724354705) q[7];
cx q[6],q[7];
ry(-0.5741552511538934) q[7];
ry(-1.5456755437501422) q[8];
cx q[7],q[8];
ry(8.270766533069464e-05) q[7];
ry(-0.012461309269978571) q[8];
cx q[7],q[8];
ry(2.4525563676408817) q[8];
ry(-2.3877495114422937) q[9];
cx q[8],q[9];
ry(-0.8901357662939163) q[8];
ry(-0.05728885541042875) q[9];
cx q[8],q[9];
ry(0.615944998667837) q[9];
ry(-0.07733549775798121) q[10];
cx q[9],q[10];
ry(-2.2231791581691978) q[9];
ry(1.1672230451544594) q[10];
cx q[9],q[10];
ry(1.2424199476672122) q[10];
ry(-2.7862052250617446) q[11];
cx q[10],q[11];
ry(1.8136756989161966) q[10];
ry(-2.802804015579294) q[11];
cx q[10],q[11];
ry(1.779376500984725) q[0];
ry(1.5020693864946661) q[1];
cx q[0],q[1];
ry(-1.4677495428876863) q[0];
ry(-2.544853953195581) q[1];
cx q[0],q[1];
ry(-2.313093936544522) q[1];
ry(-0.5969639285366712) q[2];
cx q[1],q[2];
ry(2.257690070229012) q[1];
ry(-1.7090617379943662) q[2];
cx q[1],q[2];
ry(-1.7716233491880136) q[2];
ry(1.457878710244973) q[3];
cx q[2],q[3];
ry(-2.015714330479902) q[2];
ry(2.2497171299048238) q[3];
cx q[2],q[3];
ry(3.0837637391785604) q[3];
ry(0.8764590624621099) q[4];
cx q[3],q[4];
ry(0.49571364239017246) q[3];
ry(2.959802836025152) q[4];
cx q[3],q[4];
ry(0.6568659318523764) q[4];
ry(-1.9574011249176637) q[5];
cx q[4],q[5];
ry(0.0005540247704346749) q[4];
ry(2.7210618635434867) q[5];
cx q[4],q[5];
ry(0.2059824922768403) q[5];
ry(-1.5083106500893826) q[6];
cx q[5],q[6];
ry(-0.2370046777037853) q[5];
ry(0.00032256752976689285) q[6];
cx q[5],q[6];
ry(-2.7558807264490186) q[6];
ry(1.6440541645217348) q[7];
cx q[6],q[7];
ry(-0.7788213099466449) q[6];
ry(-1.486612483762534) q[7];
cx q[6],q[7];
ry(-1.282166725490872) q[7];
ry(1.7568186354211621) q[8];
cx q[7],q[8];
ry(-1.9736599806918775e-05) q[7];
ry(-0.00872722586420327) q[8];
cx q[7],q[8];
ry(-0.29146867928732156) q[8];
ry(1.1585517624310684) q[9];
cx q[8],q[9];
ry(1.6918482745199759) q[8];
ry(0.0015131325512447447) q[9];
cx q[8],q[9];
ry(3.1243659961838577) q[9];
ry(-0.8092650162217321) q[10];
cx q[9],q[10];
ry(1.3712762241449834) q[9];
ry(2.1224378063417326) q[10];
cx q[9],q[10];
ry(-1.1873058907572256) q[10];
ry(-2.153914057921207) q[11];
cx q[10],q[11];
ry(2.3660211999277183) q[10];
ry(1.7025689378481863) q[11];
cx q[10],q[11];
ry(2.2428303716271967) q[0];
ry(-1.580405433148914) q[1];
cx q[0],q[1];
ry(-0.9431451073102821) q[0];
ry(-0.35050683072161787) q[1];
cx q[0],q[1];
ry(1.3573608928004293) q[1];
ry(-1.0293464292918202) q[2];
cx q[1],q[2];
ry(0.8858844352259947) q[1];
ry(-0.011246456755708678) q[2];
cx q[1],q[2];
ry(-0.8149128158008034) q[2];
ry(3.0787716714204905) q[3];
cx q[2],q[3];
ry(-0.7763646426831056) q[2];
ry(-0.4089576016990995) q[3];
cx q[2],q[3];
ry(-2.640213192467313) q[3];
ry(2.192654496225465) q[4];
cx q[3],q[4];
ry(0.14699604065609373) q[3];
ry(0.3429826739056461) q[4];
cx q[3],q[4];
ry(-0.5576154727689122) q[4];
ry(1.02448566459399) q[5];
cx q[4],q[5];
ry(0.6088626414801375) q[4];
ry(-0.5520636245275119) q[5];
cx q[4],q[5];
ry(0.6548400107776606) q[5];
ry(3.076047492971489) q[6];
cx q[5],q[6];
ry(-0.8785502754512962) q[5];
ry(0.011249028597593913) q[6];
cx q[5],q[6];
ry(2.147251022835552) q[6];
ry(-1.5480871066048705) q[7];
cx q[6],q[7];
ry(-0.7804777589969196) q[6];
ry(3.1313289744918005) q[7];
cx q[6],q[7];
ry(-2.4526643578682106) q[7];
ry(-0.7525516802478662) q[8];
cx q[7],q[8];
ry(3.1414891866509835) q[7];
ry(3.141513067417984) q[8];
cx q[7],q[8];
ry(-0.14291140861012064) q[8];
ry(1.732825323427108) q[9];
cx q[8],q[9];
ry(1.8890665544472256) q[8];
ry(2.958612359572937) q[9];
cx q[8],q[9];
ry(0.15103015097910172) q[9];
ry(0.6948580710056588) q[10];
cx q[9],q[10];
ry(0.754319462271807) q[9];
ry(1.8882502060539874) q[10];
cx q[9],q[10];
ry(0.48068406062190894) q[10];
ry(-1.9578083969237674) q[11];
cx q[10],q[11];
ry(-3.1228060280010266) q[10];
ry(-0.6777476221723331) q[11];
cx q[10],q[11];
ry(-1.9799711493716259) q[0];
ry(1.9289381936882513) q[1];
cx q[0],q[1];
ry(2.6985159285252087) q[0];
ry(2.6677626988043723) q[1];
cx q[0],q[1];
ry(2.8181495155643397) q[1];
ry(-3.0656545604553465) q[2];
cx q[1],q[2];
ry(-1.0941468660569909) q[1];
ry(-1.9646909308065885) q[2];
cx q[1],q[2];
ry(-2.6259757423463506) q[2];
ry(2.402081847011252) q[3];
cx q[2],q[3];
ry(-1.352555405492874) q[2];
ry(1.7033245023635761) q[3];
cx q[2],q[3];
ry(-0.537861189242844) q[3];
ry(1.628574530579904) q[4];
cx q[3],q[4];
ry(-2.559880147962133) q[3];
ry(0.5717110188962655) q[4];
cx q[3],q[4];
ry(-0.684735560602947) q[4];
ry(2.7042029525905042) q[5];
cx q[4],q[5];
ry(-0.919131538960487) q[4];
ry(-1.8143701377242536) q[5];
cx q[4],q[5];
ry(0.5280821337387716) q[5];
ry(0.9003636247790929) q[6];
cx q[5],q[6];
ry(-1.7976312432881434) q[5];
ry(0.7122662350578749) q[6];
cx q[5],q[6];
ry(1.6510538182557166) q[6];
ry(0.6931984137293743) q[7];
cx q[6],q[7];
ry(1.636145422814879) q[6];
ry(-4.9104260238408415e-05) q[7];
cx q[6],q[7];
ry(2.547961314114101) q[7];
ry(-1.9916562317530409) q[8];
cx q[7],q[8];
ry(3.1414823553031512) q[7];
ry(0.8333109137282646) q[8];
cx q[7],q[8];
ry(-2.760524879464292) q[8];
ry(2.622662218068888) q[9];
cx q[8],q[9];
ry(-2.1711069536997036) q[8];
ry(0.00015490037945564694) q[9];
cx q[8],q[9];
ry(2.5100499913124117) q[9];
ry(-1.867758299021662) q[10];
cx q[9],q[10];
ry(2.6417898559834807) q[9];
ry(-0.8862253244051095) q[10];
cx q[9],q[10];
ry(1.1925496558925806) q[10];
ry(0.5939894604780118) q[11];
cx q[10],q[11];
ry(2.6090315988501183) q[10];
ry(3.0624576400017998) q[11];
cx q[10],q[11];
ry(-1.0881943330043258) q[0];
ry(-1.603044716480668) q[1];
cx q[0],q[1];
ry(0.5803261490321772) q[0];
ry(0.13351335291847968) q[1];
cx q[0],q[1];
ry(0.09045773828689144) q[1];
ry(-0.7965664792394698) q[2];
cx q[1],q[2];
ry(-1.8827341730499456) q[1];
ry(0.5530968603977829) q[2];
cx q[1],q[2];
ry(-1.4257297735096028) q[2];
ry(0.3844569871671234) q[3];
cx q[2],q[3];
ry(0.8053566599813777) q[2];
ry(-0.10303647923518522) q[3];
cx q[2],q[3];
ry(-1.799231771496151) q[3];
ry(-0.7341848938375826) q[4];
cx q[3],q[4];
ry(-0.4077257664554066) q[3];
ry(-1.587659450741203) q[4];
cx q[3],q[4];
ry(0.18796175064575357) q[4];
ry(-1.9957102930234365) q[5];
cx q[4],q[5];
ry(0.36344834379568436) q[4];
ry(2.6004746829100474) q[5];
cx q[4],q[5];
ry(2.03709622551567) q[5];
ry(0.8292608057306552) q[6];
cx q[5],q[6];
ry(1.6274728269985133) q[5];
ry(0.7211824284194561) q[6];
cx q[5],q[6];
ry(0.9052925092586983) q[6];
ry(1.555330314077174) q[7];
cx q[6],q[7];
ry(-3.140607440056342) q[6];
ry(3.141584072129888) q[7];
cx q[6],q[7];
ry(1.7536878898659793) q[7];
ry(1.3734333997900072) q[8];
cx q[7],q[8];
ry(-3.141580312080323) q[7];
ry(2.2008826250489024) q[8];
cx q[7],q[8];
ry(-3.033076784024818) q[8];
ry(1.537880766382452) q[9];
cx q[8],q[9];
ry(-0.5725280570894168) q[8];
ry(3.133133444809104) q[9];
cx q[8],q[9];
ry(3.116342758676598) q[9];
ry(1.3408568636385945) q[10];
cx q[9],q[10];
ry(1.6414719883742928) q[9];
ry(1.2206794463459198) q[10];
cx q[9],q[10];
ry(1.8701181695646207) q[10];
ry(2.9907617482402) q[11];
cx q[10],q[11];
ry(1.9898345839191098) q[10];
ry(3.0411456565167203) q[11];
cx q[10],q[11];
ry(-1.3684723476454017) q[0];
ry(1.025793578597919) q[1];
cx q[0],q[1];
ry(1.8488971666255472) q[0];
ry(-1.2993749567305204) q[1];
cx q[0],q[1];
ry(3.0534324809353155) q[1];
ry(-0.42902356199782954) q[2];
cx q[1],q[2];
ry(-1.625024136949574) q[1];
ry(-0.9416837922183907) q[2];
cx q[1],q[2];
ry(-1.5882315787169672) q[2];
ry(0.27533788602775733) q[3];
cx q[2],q[3];
ry(-0.03170787546935517) q[2];
ry(3.105253012622793) q[3];
cx q[2],q[3];
ry(2.4243631659579443) q[3];
ry(0.6053742638342386) q[4];
cx q[3],q[4];
ry(3.078720693529598) q[3];
ry(3.084668769930729) q[4];
cx q[3],q[4];
ry(-2.6249029124375536) q[4];
ry(-1.386439329306482) q[5];
cx q[4],q[5];
ry(-3.124663807450523) q[4];
ry(-1.8489967775511813) q[5];
cx q[4],q[5];
ry(0.36128304374994435) q[5];
ry(0.9381088844976383) q[6];
cx q[5],q[6];
ry(-0.7908231364191618) q[5];
ry(0.5057982771276849) q[6];
cx q[5],q[6];
ry(-1.2803128982577114) q[6];
ry(-1.2309034936069332) q[7];
cx q[6],q[7];
ry(-8.024811741080652e-05) q[6];
ry(-0.0003648707132777318) q[7];
cx q[6],q[7];
ry(1.9490278026561407) q[7];
ry(-2.5464677523778385) q[8];
cx q[7],q[8];
ry(8.325890331519234e-05) q[7];
ry(0.011140247218648883) q[8];
cx q[7],q[8];
ry(-2.154221248769789) q[8];
ry(2.576554595562843) q[9];
cx q[8],q[9];
ry(0.3258951476400185) q[8];
ry(-2.9317298468015505) q[9];
cx q[8],q[9];
ry(-1.92038466790052) q[9];
ry(-3.0706664763651568) q[10];
cx q[9],q[10];
ry(-1.4498356103566612) q[9];
ry(0.2487113253318309) q[10];
cx q[9],q[10];
ry(-0.7924304200011753) q[10];
ry(2.1373110189904114) q[11];
cx q[10],q[11];
ry(-1.0457087958700964) q[10];
ry(0.7529758308937362) q[11];
cx q[10],q[11];
ry(2.638277290149947) q[0];
ry(-2.8107796275339396) q[1];
cx q[0],q[1];
ry(-2.476896979147198) q[0];
ry(1.7354298042090988) q[1];
cx q[0],q[1];
ry(-2.4778623815641883) q[1];
ry(0.49605121058037116) q[2];
cx q[1],q[2];
ry(-0.7996601355998338) q[1];
ry(-3.0482507241099865) q[2];
cx q[1],q[2];
ry(0.04909684569024452) q[2];
ry(-1.7480731711673347) q[3];
cx q[2],q[3];
ry(1.8548843349118207) q[2];
ry(0.09739113105254772) q[3];
cx q[2],q[3];
ry(-1.774631848603203) q[3];
ry(-2.4495758343627636) q[4];
cx q[3],q[4];
ry(0.00844468207139415) q[3];
ry(0.6739671924716203) q[4];
cx q[3],q[4];
ry(-1.2743043407340244) q[4];
ry(2.481311781565094) q[5];
cx q[4],q[5];
ry(0.13969630898258512) q[4];
ry(0.021378364383304448) q[5];
cx q[4],q[5];
ry(2.3452464738605303) q[5];
ry(1.253526200297377) q[6];
cx q[5],q[6];
ry(2.5391724860099147) q[5];
ry(2.576776489897448) q[6];
cx q[5],q[6];
ry(-2.3597636159184128) q[6];
ry(-0.3315610038821209) q[7];
cx q[6],q[7];
ry(0.09665557790901547) q[6];
ry(-0.0004887416170953403) q[7];
cx q[6],q[7];
ry(2.685663541385943) q[7];
ry(-1.0357231697415523) q[8];
cx q[7],q[8];
ry(-0.00021590573900605412) q[7];
ry(3.138854532362975) q[8];
cx q[7],q[8];
ry(-2.504073336565848) q[8];
ry(-1.8806789262007086) q[9];
cx q[8],q[9];
ry(-0.7054104409256343) q[8];
ry(-0.6081871050612246) q[9];
cx q[8],q[9];
ry(0.6733667916688073) q[9];
ry(-0.43973368373824234) q[10];
cx q[9],q[10];
ry(-2.41051923112344) q[9];
ry(2.372813489644789) q[10];
cx q[9],q[10];
ry(-1.176143915289715) q[10];
ry(-0.7665399453209645) q[11];
cx q[10],q[11];
ry(1.1001467317907787) q[10];
ry(-0.817116751344181) q[11];
cx q[10],q[11];
ry(-2.4083545678466094) q[0];
ry(1.3800410046308356) q[1];
cx q[0],q[1];
ry(1.573975218366753) q[0];
ry(1.501773963550183) q[1];
cx q[0],q[1];
ry(-1.2153471848920057) q[1];
ry(0.8847052604557735) q[2];
cx q[1],q[2];
ry(3.1209018439432583) q[1];
ry(1.46699093344463) q[2];
cx q[1],q[2];
ry(-2.0346918007647625) q[2];
ry(-0.823670231646329) q[3];
cx q[2],q[3];
ry(1.4137529943460265) q[2];
ry(-3.1003989018244784) q[3];
cx q[2],q[3];
ry(2.25242593780315) q[3];
ry(2.1802616844630345) q[4];
cx q[3],q[4];
ry(-2.2559074172748215) q[3];
ry(-0.9498780457835655) q[4];
cx q[3],q[4];
ry(-0.8701292928438367) q[4];
ry(0.6394243389558597) q[5];
cx q[4],q[5];
ry(-1.787412367790105) q[4];
ry(-2.9934648261129735) q[5];
cx q[4],q[5];
ry(-2.278561736588235) q[5];
ry(-1.022369378408384) q[6];
cx q[5],q[6];
ry(0.006763629652247971) q[5];
ry(2.7361593351882934) q[6];
cx q[5],q[6];
ry(-3.115355981745218) q[6];
ry(-0.732865817865541) q[7];
cx q[6],q[7];
ry(-1.6342422613601677) q[6];
ry(4.028021239886297e-05) q[7];
cx q[6],q[7];
ry(0.44392723389437455) q[7];
ry(-0.7461286085891751) q[8];
cx q[7],q[8];
ry(-0.00032862980401048394) q[7];
ry(-3.1415739351769356) q[8];
cx q[7],q[8];
ry(-0.45749089340734145) q[8];
ry(0.7433311179048304) q[9];
cx q[8],q[9];
ry(0.12816079947939005) q[8];
ry(-0.4653627119893891) q[9];
cx q[8],q[9];
ry(0.7158065615674871) q[9];
ry(1.8751469435474781) q[10];
cx q[9],q[10];
ry(-2.946474858677883) q[9];
ry(2.3370398886252133) q[10];
cx q[9],q[10];
ry(0.4142929270858336) q[10];
ry(0.9637641143167077) q[11];
cx q[10],q[11];
ry(-2.1546539189374645) q[10];
ry(1.782276903655775) q[11];
cx q[10],q[11];
ry(1.3040255181146359) q[0];
ry(0.7450700341949329) q[1];
cx q[0],q[1];
ry(2.0834826266603628) q[0];
ry(1.704333696492208) q[1];
cx q[0],q[1];
ry(-2.363038975243359) q[1];
ry(-1.0611293808397861) q[2];
cx q[1],q[2];
ry(2.586150801007813) q[1];
ry(-1.8272933093842847) q[2];
cx q[1],q[2];
ry(1.9451006038988254) q[2];
ry(1.9901318151923946) q[3];
cx q[2],q[3];
ry(-3.0101788508771596) q[2];
ry(2.460738184921832) q[3];
cx q[2],q[3];
ry(1.0961843345033113) q[3];
ry(1.980910736191106) q[4];
cx q[3],q[4];
ry(-0.8222362036306654) q[3];
ry(2.905164797658224) q[4];
cx q[3],q[4];
ry(-1.6850403978268904) q[4];
ry(-2.124452664070354) q[5];
cx q[4],q[5];
ry(-1.810928666295884) q[4];
ry(-0.08122803901966327) q[5];
cx q[4],q[5];
ry(-0.01896465999612218) q[5];
ry(-2.0166289632382153) q[6];
cx q[5],q[6];
ry(1.4187847208597857) q[5];
ry(-0.9844742084773144) q[6];
cx q[5],q[6];
ry(2.0378636114996045) q[6];
ry(1.2966282905170363) q[7];
cx q[6],q[7];
ry(-0.2570553280296295) q[6];
ry(3.0388648047417877) q[7];
cx q[6],q[7];
ry(-1.36410753782899) q[7];
ry(-2.6880007473145135) q[8];
cx q[7],q[8];
ry(-3.141518645885275) q[7];
ry(-0.0005829144665741026) q[8];
cx q[7],q[8];
ry(-1.5156729268287386) q[8];
ry(1.0540716354048092) q[9];
cx q[8],q[9];
ry(-0.179743733844818) q[8];
ry(0.013838649821441784) q[9];
cx q[8],q[9];
ry(1.638894129909354) q[9];
ry(-1.4943003243410722) q[10];
cx q[9],q[10];
ry(-2.190269150226132) q[9];
ry(2.5604022799027906) q[10];
cx q[9],q[10];
ry(2.1298721559438656) q[10];
ry(-1.4812219366518153) q[11];
cx q[10],q[11];
ry(-2.714969796448452) q[10];
ry(-1.1636981923044598) q[11];
cx q[10],q[11];
ry(1.0267114596122813) q[0];
ry(-3.055918838842725) q[1];
cx q[0],q[1];
ry(0.029652021685746658) q[0];
ry(1.1837693177787985) q[1];
cx q[0],q[1];
ry(2.1754626749114756) q[1];
ry(-1.4910112393104227) q[2];
cx q[1],q[2];
ry(0.5354770829578355) q[1];
ry(-1.9905267652674796) q[2];
cx q[1],q[2];
ry(-2.4112840638297923) q[2];
ry(-0.5114270283684164) q[3];
cx q[2],q[3];
ry(-2.694471572255154) q[2];
ry(-0.38155152184137986) q[3];
cx q[2],q[3];
ry(-1.33623334048489) q[3];
ry(1.0000489336645457) q[4];
cx q[3],q[4];
ry(0.5762214828754582) q[3];
ry(2.235067001589875) q[4];
cx q[3],q[4];
ry(3.117561636756773) q[4];
ry(2.491440479795931) q[5];
cx q[4],q[5];
ry(0.00676985221229387) q[4];
ry(-1.6994314747643018) q[5];
cx q[4],q[5];
ry(1.1729967330652784) q[5];
ry(2.6989584253854377) q[6];
cx q[5],q[6];
ry(0.04876155617671223) q[5];
ry(3.136513609465328) q[6];
cx q[5],q[6];
ry(-0.33658596639525895) q[6];
ry(-1.6750510862908503) q[7];
cx q[6],q[7];
ry(0.8626559308933093) q[6];
ry(-0.030455602647363904) q[7];
cx q[6],q[7];
ry(-1.476172340598639) q[7];
ry(1.3050356737468602) q[8];
cx q[7],q[8];
ry(-0.0002796630306777955) q[7];
ry(3.1408445960241016) q[8];
cx q[7],q[8];
ry(-2.677518270103586) q[8];
ry(1.2052668873812604) q[9];
cx q[8],q[9];
ry(-1.6254346521974627) q[8];
ry(0.3800009852451283) q[9];
cx q[8],q[9];
ry(-1.6102653404801295) q[9];
ry(-0.32679970253869844) q[10];
cx q[9],q[10];
ry(-3.028299363794823) q[9];
ry(-1.6571793332458675) q[10];
cx q[9],q[10];
ry(1.877383976425569) q[10];
ry(2.239486801922496) q[11];
cx q[10],q[11];
ry(2.493612190904462) q[10];
ry(1.8244669101719788) q[11];
cx q[10],q[11];
ry(-1.5281323793996773) q[0];
ry(2.881527998897075) q[1];
cx q[0],q[1];
ry(-0.7165566914560453) q[0];
ry(0.042239368291066094) q[1];
cx q[0],q[1];
ry(-1.697728970148843) q[1];
ry(2.2943376205417345) q[2];
cx q[1],q[2];
ry(3.100207063963358) q[1];
ry(0.8892010290313783) q[2];
cx q[1],q[2];
ry(-2.506244336802167) q[2];
ry(1.3201620176280446) q[3];
cx q[2],q[3];
ry(-2.509513130308021) q[2];
ry(-0.6507685605569895) q[3];
cx q[2],q[3];
ry(-1.7083397779546714) q[3];
ry(3.004553515320163) q[4];
cx q[3],q[4];
ry(1.115575278136828) q[3];
ry(-0.818699317735448) q[4];
cx q[3],q[4];
ry(-2.6179351041598573) q[4];
ry(-1.7634226859599536) q[5];
cx q[4],q[5];
ry(1.03436903165756) q[4];
ry(-1.9572908040227244) q[5];
cx q[4],q[5];
ry(1.7139794376324646) q[5];
ry(0.5015329677949653) q[6];
cx q[5],q[6];
ry(-0.006571666165427814) q[5];
ry(3.1391663692280978) q[6];
cx q[5],q[6];
ry(0.11077971646884421) q[6];
ry(1.9135853096705842) q[7];
cx q[6],q[7];
ry(2.304255108829722) q[6];
ry(-3.0125761687774597) q[7];
cx q[6],q[7];
ry(0.6708476791046678) q[7];
ry(-0.7278041220080921) q[8];
cx q[7],q[8];
ry(-9.517956401960959e-05) q[7];
ry(-0.002350706930696056) q[8];
cx q[7],q[8];
ry(-2.772016105588787) q[8];
ry(1.2935372755337218) q[9];
cx q[8],q[9];
ry(2.6290417006182043) q[8];
ry(-1.1768487302414536) q[9];
cx q[8],q[9];
ry(2.0907178225659733) q[9];
ry(0.08715109180753977) q[10];
cx q[9],q[10];
ry(-2.21624692804878) q[9];
ry(-3.0507864532979605) q[10];
cx q[9],q[10];
ry(-0.8846240786081166) q[10];
ry(-2.227798710150227) q[11];
cx q[10],q[11];
ry(-0.12648289140884136) q[10];
ry(-1.8895204454982035) q[11];
cx q[10],q[11];
ry(-0.9745622695326348) q[0];
ry(1.234363806553153) q[1];
cx q[0],q[1];
ry(-2.894332103898738) q[0];
ry(0.7681883560456382) q[1];
cx q[0],q[1];
ry(0.5187464643661244) q[1];
ry(0.7473204632245662) q[2];
cx q[1],q[2];
ry(2.3724487514372257) q[1];
ry(0.03479214801395418) q[2];
cx q[1],q[2];
ry(1.9319419509834683) q[2];
ry(1.2425116214914045) q[3];
cx q[2],q[3];
ry(3.1187315867515992) q[2];
ry(0.06095381264652313) q[3];
cx q[2],q[3];
ry(3.136108991738165) q[3];
ry(-2.9929783562636185) q[4];
cx q[3],q[4];
ry(0.01626859600439316) q[3];
ry(-1.4447863058516217) q[4];
cx q[3],q[4];
ry(-1.3585663724235548) q[4];
ry(-1.3762886541970119) q[5];
cx q[4],q[5];
ry(2.0681821225467436) q[4];
ry(-2.1552890984818798) q[5];
cx q[4],q[5];
ry(2.7318733563859596) q[5];
ry(-2.2052326407483833) q[6];
cx q[5],q[6];
ry(0.850164517179494) q[5];
ry(-0.3237385170025817) q[6];
cx q[5],q[6];
ry(-1.3333646307241704) q[6];
ry(-1.6366467677823944) q[7];
cx q[6],q[7];
ry(1.3748887094588624) q[6];
ry(1.8953101771196312) q[7];
cx q[6],q[7];
ry(-0.26475655308265006) q[7];
ry(2.649971946221756) q[8];
cx q[7],q[8];
ry(3.141017459831129) q[7];
ry(-3.141410536164096) q[8];
cx q[7],q[8];
ry(-0.7245857304803112) q[8];
ry(-0.1219255677502329) q[9];
cx q[8],q[9];
ry(-0.40168910713432937) q[8];
ry(0.4789091219748185) q[9];
cx q[8],q[9];
ry(-3.068907511273649) q[9];
ry(2.146905180946077) q[10];
cx q[9],q[10];
ry(1.752999785029603) q[9];
ry(-2.9342243805843844) q[10];
cx q[9],q[10];
ry(0.19769870588876132) q[10];
ry(-0.6036788855783752) q[11];
cx q[10],q[11];
ry(2.624888208321671) q[10];
ry(0.10306337808419444) q[11];
cx q[10],q[11];
ry(-0.38299675112698184) q[0];
ry(1.9163051955026493) q[1];
cx q[0],q[1];
ry(0.1809602099135761) q[0];
ry(-0.05912284133762835) q[1];
cx q[0],q[1];
ry(-0.26815626845383367) q[1];
ry(-1.843064780545407) q[2];
cx q[1],q[2];
ry(-0.6621788215443863) q[1];
ry(-2.9898259422685323) q[2];
cx q[1],q[2];
ry(2.8080391364133357) q[2];
ry(0.25392199888432554) q[3];
cx q[2],q[3];
ry(-0.0693580319263063) q[2];
ry(-3.1249461753895673) q[3];
cx q[2],q[3];
ry(2.9742668430197456) q[3];
ry(2.8678582889409836) q[4];
cx q[3],q[4];
ry(3.0537998966202715) q[3];
ry(-3.0137488629439706) q[4];
cx q[3],q[4];
ry(0.985849818452321) q[4];
ry(-0.16238432552350326) q[5];
cx q[4],q[5];
ry(2.3312195570498786) q[4];
ry(-0.20581014885031645) q[5];
cx q[4],q[5];
ry(0.1028784866067971) q[5];
ry(0.2259073422634474) q[6];
cx q[5],q[6];
ry(0.003598447624297019) q[5];
ry(3.140506358392542) q[6];
cx q[5],q[6];
ry(-0.2278283722425434) q[6];
ry(0.7818751079164742) q[7];
cx q[6],q[7];
ry(-3.1374997852390036) q[6];
ry(1.2451039468016243) q[7];
cx q[6],q[7];
ry(-1.7419255757343108) q[7];
ry(-0.8214647203580151) q[8];
cx q[7],q[8];
ry(-3.1400326526993942) q[7];
ry(0.0026895263378906107) q[8];
cx q[7],q[8];
ry(-2.3352293824956614) q[8];
ry(0.23711774441088998) q[9];
cx q[8],q[9];
ry(-1.6895427996819095) q[8];
ry(0.37813145437523854) q[9];
cx q[8],q[9];
ry(-2.856198691159167) q[9];
ry(1.9981342900119257) q[10];
cx q[9],q[10];
ry(-1.4834797851500978) q[9];
ry(0.5578505511839926) q[10];
cx q[9],q[10];
ry(0.3154846513391582) q[10];
ry(-1.310100387290543) q[11];
cx q[10],q[11];
ry(-1.4822795976060243) q[10];
ry(-0.4263105015019253) q[11];
cx q[10],q[11];
ry(0.6059808587919515) q[0];
ry(1.1706925602078504) q[1];
cx q[0],q[1];
ry(-2.2296109963664232) q[0];
ry(-2.5612493860822982) q[1];
cx q[0],q[1];
ry(-2.5724867796573303) q[1];
ry(0.6188315712472597) q[2];
cx q[1],q[2];
ry(-0.000542501512369657) q[1];
ry(-0.2485197153274816) q[2];
cx q[1],q[2];
ry(1.888661142428559) q[2];
ry(-0.25026724160440406) q[3];
cx q[2],q[3];
ry(-0.5651529138413114) q[2];
ry(2.4907761327628415) q[3];
cx q[2],q[3];
ry(1.9295939984484054) q[3];
ry(1.0687227936926629) q[4];
cx q[3],q[4];
ry(3.1050739009753934) q[3];
ry(0.9152927933033126) q[4];
cx q[3],q[4];
ry(-2.9165394561697937) q[4];
ry(-3.0806061018562945) q[5];
cx q[4],q[5];
ry(2.3045640637925007) q[4];
ry(-0.2556272483405202) q[5];
cx q[4],q[5];
ry(0.5028419949021181) q[5];
ry(3.016782834107425) q[6];
cx q[5],q[6];
ry(0.041202816849565026) q[5];
ry(-2.859603029499795) q[6];
cx q[5],q[6];
ry(-2.2372530153843373) q[6];
ry(0.8779941867011926) q[7];
cx q[6],q[7];
ry(-0.14114458211330216) q[6];
ry(2.986673887817126) q[7];
cx q[6],q[7];
ry(2.9633334494691734) q[7];
ry(1.4207988458141803) q[8];
cx q[7],q[8];
ry(-0.5655022137147955) q[7];
ry(-0.0005432981590312025) q[8];
cx q[7],q[8];
ry(-1.575710267963478) q[8];
ry(0.9244140701362582) q[9];
cx q[8],q[9];
ry(3.137302708383636) q[8];
ry(0.620874196377848) q[9];
cx q[8],q[9];
ry(-1.3913648698105874) q[9];
ry(2.249104837955172) q[10];
cx q[9],q[10];
ry(-3.139376483141004) q[9];
ry(-1.3086371487409663) q[10];
cx q[9],q[10];
ry(2.7807946025367523) q[10];
ry(1.1419151767263767) q[11];
cx q[10],q[11];
ry(2.643304190940299) q[10];
ry(3.065707368540754) q[11];
cx q[10],q[11];
ry(-2.188124481678864) q[0];
ry(-2.40517047722271) q[1];
cx q[0],q[1];
ry(-1.2627725968001133) q[0];
ry(1.448451241783049) q[1];
cx q[0],q[1];
ry(2.579027749777241) q[1];
ry(-0.7752059932883144) q[2];
cx q[1],q[2];
ry(2.355262070954773) q[1];
ry(1.676991269679887) q[2];
cx q[1],q[2];
ry(-1.5991041378115614) q[2];
ry(3.052664523011417) q[3];
cx q[2],q[3];
ry(0.0183407780396303) q[2];
ry(-0.45783188062839525) q[3];
cx q[2],q[3];
ry(-0.09264256268450097) q[3];
ry(-2.2689550118448985) q[4];
cx q[3],q[4];
ry(-0.25582683221542624) q[3];
ry(-0.5219504031502176) q[4];
cx q[3],q[4];
ry(2.6904653015938234) q[4];
ry(2.806039865917984) q[5];
cx q[4],q[5];
ry(-0.22720047111674968) q[4];
ry(-2.336391668314064) q[5];
cx q[4],q[5];
ry(2.1128071970658757) q[5];
ry(-0.8105375910551863) q[6];
cx q[5],q[6];
ry(3.1387924509861924) q[5];
ry(-3.1364612279920157) q[6];
cx q[5],q[6];
ry(-0.6628296305253929) q[6];
ry(1.266813616167668) q[7];
cx q[6],q[7];
ry(3.1403798961543123) q[6];
ry(-2.999629525588408) q[7];
cx q[6],q[7];
ry(0.7060878614826908) q[7];
ry(1.5750346880820656) q[8];
cx q[7],q[8];
ry(-2.5760949994545976) q[7];
ry(-3.1412262134145634) q[8];
cx q[7],q[8];
ry(-0.8776871495903759) q[8];
ry(2.5204534714736107) q[9];
cx q[8],q[9];
ry(2.025603050447696) q[8];
ry(-2.4230823397561263) q[9];
cx q[8],q[9];
ry(-1.246718368488906) q[9];
ry(-0.35253178608579727) q[10];
cx q[9],q[10];
ry(0.022958087054434806) q[9];
ry(2.6862297499624903) q[10];
cx q[9],q[10];
ry(0.4107559496737236) q[10];
ry(-0.44748618889208647) q[11];
cx q[10],q[11];
ry(3.057087444660457) q[10];
ry(2.6712559440323234) q[11];
cx q[10],q[11];
ry(-0.14262055236209298) q[0];
ry(-1.246920577571749) q[1];
cx q[0],q[1];
ry(0.06647448494954772) q[0];
ry(2.671359885083035) q[1];
cx q[0],q[1];
ry(1.395942482685599) q[1];
ry(1.6493212158747976) q[2];
cx q[1],q[2];
ry(0.7844692932839306) q[1];
ry(-3.013487323520481) q[2];
cx q[1],q[2];
ry(0.6189160327232619) q[2];
ry(-1.7776749917531527) q[3];
cx q[2],q[3];
ry(-2.2457815766112237) q[2];
ry(1.2067046856658872) q[3];
cx q[2],q[3];
ry(-0.5583977080928395) q[3];
ry(-2.0930942465244433) q[4];
cx q[3],q[4];
ry(3.0279852896424835) q[3];
ry(2.336893004351312) q[4];
cx q[3],q[4];
ry(1.4919342585778423) q[4];
ry(2.955004260013255) q[5];
cx q[4],q[5];
ry(-0.44638359272797246) q[4];
ry(2.973478588714655) q[5];
cx q[4],q[5];
ry(2.916421142349652) q[5];
ry(2.201108933642204) q[6];
cx q[5],q[6];
ry(-3.139541532440093) q[5];
ry(-2.7699387147054773) q[6];
cx q[5],q[6];
ry(0.7783543984093475) q[6];
ry(2.9918588514431868) q[7];
cx q[6],q[7];
ry(-0.9956627539475624) q[6];
ry(-1.933386706423351) q[7];
cx q[6],q[7];
ry(1.8441817016305526) q[7];
ry(-0.008551951277644623) q[8];
cx q[7],q[8];
ry(0.0006048351162366572) q[7];
ry(3.141491576979213) q[8];
cx q[7],q[8];
ry(2.007820621401125) q[8];
ry(1.6858067806144161) q[9];
cx q[8],q[9];
ry(-0.8966954789910808) q[8];
ry(-2.5851441942797795) q[9];
cx q[8],q[9];
ry(0.753358532095818) q[9];
ry(-0.6487773934991321) q[10];
cx q[9],q[10];
ry(3.0690726513816413) q[9];
ry(-2.1316475478151053) q[10];
cx q[9],q[10];
ry(2.824155707979527) q[10];
ry(0.9077258793409774) q[11];
cx q[10],q[11];
ry(1.749247905284479) q[10];
ry(0.7403224921578975) q[11];
cx q[10],q[11];
ry(-1.0945519408979285) q[0];
ry(-0.9120469499494348) q[1];
cx q[0],q[1];
ry(3.053040933064004) q[0];
ry(-2.1181148471295312) q[1];
cx q[0],q[1];
ry(-3.0698628718841388) q[1];
ry(-1.021299157661896) q[2];
cx q[1],q[2];
ry(-0.020379308374933736) q[1];
ry(0.9356603358637758) q[2];
cx q[1],q[2];
ry(0.6445840171088836) q[2];
ry(0.07113566698828495) q[3];
cx q[2],q[3];
ry(-1.764436908108574) q[2];
ry(3.0949716870035062) q[3];
cx q[2],q[3];
ry(-0.6613179177872951) q[3];
ry(2.0562179254962283) q[4];
cx q[3],q[4];
ry(-1.8965520597260863) q[3];
ry(1.9753985906003309) q[4];
cx q[3],q[4];
ry(0.6023390059877461) q[4];
ry(1.5511292489384143) q[5];
cx q[4],q[5];
ry(-0.3458186324652302) q[4];
ry(0.09132278277486616) q[5];
cx q[4],q[5];
ry(-1.4671924613565641) q[5];
ry(-0.930983281455659) q[6];
cx q[5],q[6];
ry(-3.1401142938603903) q[5];
ry(0.006903597657603734) q[6];
cx q[5],q[6];
ry(1.0733137294380022) q[6];
ry(2.5239667154811243) q[7];
cx q[6],q[7];
ry(0.31557176198634307) q[6];
ry(-3.124118651911901) q[7];
cx q[6],q[7];
ry(0.5596174110502043) q[7];
ry(0.017255606123596227) q[8];
cx q[7],q[8];
ry(-0.011661335920171823) q[7];
ry(0.003859164698192267) q[8];
cx q[7],q[8];
ry(-1.3793428288125118) q[8];
ry(-0.013989884056949897) q[9];
cx q[8],q[9];
ry(0.15281075126992186) q[8];
ry(-2.7669807997074174) q[9];
cx q[8],q[9];
ry(2.505849594745074) q[9];
ry(1.9443495333755374) q[10];
cx q[9],q[10];
ry(1.3701226938081537) q[9];
ry(-0.7633729268973656) q[10];
cx q[9],q[10];
ry(2.1925479550674964) q[10];
ry(-2.7794182733182144) q[11];
cx q[10],q[11];
ry(-2.7767968903972857) q[10];
ry(-2.008193407685358) q[11];
cx q[10],q[11];
ry(2.3876393498334405) q[0];
ry(-2.515289137102016) q[1];
cx q[0],q[1];
ry(-1.8051954065906237) q[0];
ry(-2.268443533318485) q[1];
cx q[0],q[1];
ry(-1.822066680335122) q[1];
ry(-2.0634830072937236) q[2];
cx q[1],q[2];
ry(-0.06245454408093831) q[1];
ry(0.12087490586083714) q[2];
cx q[1],q[2];
ry(2.9852214646638475) q[2];
ry(1.0738704212507706) q[3];
cx q[2],q[3];
ry(-0.07025395329752931) q[2];
ry(0.8639294386050578) q[3];
cx q[2],q[3];
ry(-3.016374586411122) q[3];
ry(2.6286037807185445) q[4];
cx q[3],q[4];
ry(1.6915411382855563) q[3];
ry(-0.6596801341801821) q[4];
cx q[3],q[4];
ry(-1.560427392167253) q[4];
ry(1.1075553072568927) q[5];
cx q[4],q[5];
ry(-2.0413531821027004) q[4];
ry(2.8814273728522024) q[5];
cx q[4],q[5];
ry(0.6655167382381981) q[5];
ry(1.0304452102504307) q[6];
cx q[5],q[6];
ry(-3.141218882897463) q[5];
ry(3.1398882592716353) q[6];
cx q[5],q[6];
ry(2.2429314548449666) q[6];
ry(1.8642228399547607) q[7];
cx q[6],q[7];
ry(-2.109810511863037) q[6];
ry(0.8313904986374929) q[7];
cx q[6],q[7];
ry(1.8772410489275004) q[7];
ry(-0.9428918039534278) q[8];
cx q[7],q[8];
ry(-2.75755461006657) q[7];
ry(0.19977604703482044) q[8];
cx q[7],q[8];
ry(0.4145009391001153) q[8];
ry(0.12543149715979707) q[9];
cx q[8],q[9];
ry(3.117752126567612) q[8];
ry(-0.003951039976549511) q[9];
cx q[8],q[9];
ry(-1.219171862802675) q[9];
ry(1.7465622398268326) q[10];
cx q[9],q[10];
ry(-0.6739801340117811) q[9];
ry(2.1634582049926188) q[10];
cx q[9],q[10];
ry(-1.521955793490604) q[10];
ry(2.976891374347016) q[11];
cx q[10],q[11];
ry(-1.6167534146262321) q[10];
ry(1.0831355722629539) q[11];
cx q[10],q[11];
ry(-0.5334773149910665) q[0];
ry(3.1212905103273765) q[1];
cx q[0],q[1];
ry(2.2299523379327266) q[0];
ry(-0.3121337197433668) q[1];
cx q[0],q[1];
ry(0.4910012335348392) q[1];
ry(-0.8712643308180512) q[2];
cx q[1],q[2];
ry(2.8801957906259434) q[1];
ry(3.0528541699282536) q[2];
cx q[1],q[2];
ry(1.3893809863673392) q[2];
ry(0.6005855421046241) q[3];
cx q[2],q[3];
ry(-3.0129586329026132) q[2];
ry(2.065580436326135) q[3];
cx q[2],q[3];
ry(-2.189250503373592) q[3];
ry(-1.4830820926274488) q[4];
cx q[3],q[4];
ry(-0.3263767943472305) q[3];
ry(1.0015981513970327) q[4];
cx q[3],q[4];
ry(1.2970985564769322) q[4];
ry(-1.3121929227024696) q[5];
cx q[4],q[5];
ry(-0.506582863309851) q[4];
ry(0.6297127681044123) q[5];
cx q[4],q[5];
ry(0.6450847666159722) q[5];
ry(-1.6718701487544605) q[6];
cx q[5],q[6];
ry(-2.9107857691379344) q[5];
ry(-3.0658267504142316) q[6];
cx q[5],q[6];
ry(1.0645488185428515) q[6];
ry(-1.8814467549990113) q[7];
cx q[6],q[7];
ry(3.1175005229786086) q[6];
ry(0.001539481339695097) q[7];
cx q[6],q[7];
ry(1.9255860038889632) q[7];
ry(2.798754354218614) q[8];
cx q[7],q[8];
ry(-2.5773273428513646) q[7];
ry(0.26995357263570163) q[8];
cx q[7],q[8];
ry(1.513128293690249) q[8];
ry(1.7999795635811142) q[9];
cx q[8],q[9];
ry(-2.6974732464569975) q[8];
ry(0.4511772350721437) q[9];
cx q[8],q[9];
ry(-1.3603549604347513) q[9];
ry(-1.8380634118120849) q[10];
cx q[9],q[10];
ry(1.9940625514554233) q[9];
ry(-2.677212146853916) q[10];
cx q[9],q[10];
ry(-1.1203138377444741) q[10];
ry(0.6307236915346566) q[11];
cx q[10],q[11];
ry(-1.1998312129084638) q[10];
ry(-0.4113588312345069) q[11];
cx q[10],q[11];
ry(-0.24896927003670033) q[0];
ry(-0.5727741309080416) q[1];
ry(-0.38497104307109264) q[2];
ry(2.0193858514295124) q[3];
ry(2.755718176628786) q[4];
ry(0.06048595846962693) q[5];
ry(-2.618267492421266) q[6];
ry(2.853061131723836) q[7];
ry(-2.9716091858190272e-05) q[8];
ry(3.1231819971504406) q[9];
ry(0.026038591856282753) q[10];
ry(-0.0617976935654001) q[11];