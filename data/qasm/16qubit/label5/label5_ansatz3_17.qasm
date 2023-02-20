OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
ry(1.5546704303494419) q[0];
rz(-1.6837475324499078) q[0];
ry(-2.0279666580867715) q[1];
rz(1.4905650842157674) q[1];
ry(-1.7413997026185226) q[2];
rz(3.076948103709436) q[2];
ry(0.0013973987024409925) q[3];
rz(1.326512699124252) q[3];
ry(0.0006322341858416146) q[4];
rz(-2.19424223858754) q[4];
ry(1.5672846009706198) q[5];
rz(-1.5732962121958316) q[5];
ry(-2.0231067547064825) q[6];
rz(0.14616141850824094) q[6];
ry(-1.022471635978519) q[7];
rz(-0.00827759702513653) q[7];
ry(3.139974723416932) q[8];
rz(-0.2681118126225384) q[8];
ry(-0.0651149741696711) q[9];
rz(2.0708437702853963) q[9];
ry(-3.1413713914217345) q[10];
rz(-0.31736308937634816) q[10];
ry(-1.2197099637085074) q[11];
rz(0.5617885425944581) q[11];
ry(-1.4282928513018796) q[12];
rz(-1.8871150863448438) q[12];
ry(3.0200800882918957) q[13];
rz(-2.545784656469761) q[13];
ry(2.2261197619136377) q[14];
rz(-0.437236252246446) q[14];
ry(1.713080054533999) q[15];
rz(-1.8859970246455384) q[15];
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
ry(3.0660644797027734) q[0];
rz(1.6562567553766359) q[0];
ry(-3.117077930981949) q[1];
rz(0.5237578250509403) q[1];
ry(0.21099658234814544) q[2];
rz(1.7274668738045293) q[2];
ry(-2.1660516298875256) q[3];
rz(-2.5864607830751525) q[3];
ry(0.1371871993090053) q[4];
rz(1.4635173398594485) q[4];
ry(-1.34156867078421) q[5];
rz(2.617061289745768) q[5];
ry(2.494902545366151) q[6];
rz(-1.604706817029193) q[6];
ry(-0.002881288170445241) q[7];
rz(3.080387525265136) q[7];
ry(-1.5484891204067788) q[8];
rz(-1.5591923705097233) q[8];
ry(-0.0653402598243702) q[9];
rz(-2.1459845244541773) q[9];
ry(-0.0017996029662700683) q[10];
rz(1.9510915812402656) q[10];
ry(2.79116838355149) q[11];
rz(-3.0466040449155547) q[11];
ry(0.9599296141794429) q[12];
rz(1.140954126684752) q[12];
ry(-0.007175616691883491) q[13];
rz(2.7757976446454236) q[13];
ry(2.2403970056004665) q[14];
rz(-0.6752878772960926) q[14];
ry(2.31475450449628) q[15];
rz(1.103080776519672) q[15];
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
ry(1.7550841096510856) q[0];
rz(1.643900662849369) q[0];
ry(-0.11799497527159294) q[1];
rz(-2.308549546729264) q[1];
ry(3.109094019371746) q[2];
rz(-1.9689637275353997) q[2];
ry(-0.0006493969131084559) q[3];
rz(-0.28679784894204696) q[3];
ry(-3.1412663684637434) q[4];
rz(2.5072880135871185) q[4];
ry(0.0006513287106830745) q[5];
rz(-2.6116588505749903) q[5];
ry(1.6046022413896637) q[6];
rz(-3.1329498429880513) q[6];
ry(1.8901917421952386) q[7];
rz(3.0354969886592347) q[7];
ry(1.7535762996801654) q[8];
rz(0.013466619645451918) q[8];
ry(3.1309999041737644) q[9];
rz(-2.804976441616655) q[9];
ry(-3.1413947694232727) q[10];
rz(-0.6440844816759865) q[10];
ry(2.29694444491267) q[11];
rz(0.9105766167107632) q[11];
ry(-3.1219349020727907) q[12];
rz(-0.24649229980608925) q[12];
ry(-3.071202303891915) q[13];
rz(1.7975324838051145) q[13];
ry(2.299312273476179) q[14];
rz(-1.584463936809696) q[14];
ry(0.031099980991714356) q[15];
rz(-2.4568561968288316) q[15];
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
ry(2.57074128940695) q[0];
rz(0.6486116461849578) q[0];
ry(0.05471092987157267) q[1];
rz(-0.7485111883209846) q[1];
ry(0.19966106449687868) q[2];
rz(1.6623659559846544) q[2];
ry(-2.8554172627720527) q[3];
rz(2.0961168203152827) q[3];
ry(0.0007927299053473755) q[4];
rz(-2.634166661940889) q[4];
ry(-1.4348253297844638) q[5];
rz(1.9157878932901473) q[5];
ry(1.5822202406802353) q[6];
rz(0.12002052021388643) q[6];
ry(0.013618821189819386) q[7];
rz(-3.1035659329283174) q[7];
ry(-0.7053802702113332) q[8];
rz(1.3686066411961595) q[8];
ry(-0.13164368479681077) q[9];
rz(3.0596184389272483) q[9];
ry(3.1381663832716593) q[10];
rz(-1.3161692425384848) q[10];
ry(-0.2867728791427586) q[11];
rz(-0.5457222971822038) q[11];
ry(-0.6067406562546465) q[12];
rz(-2.641328998699376) q[12];
ry(-1.4643665940101938) q[13];
rz(-2.5756816739360247) q[13];
ry(-0.5543589175407364) q[14];
rz(-2.6830075157210027) q[14];
ry(-1.0023701688799096) q[15];
rz(2.9564357542989614) q[15];
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
ry(-0.20546662326788218) q[0];
rz(-0.8510277045501216) q[0];
ry(3.023855959695827) q[1];
rz(-0.22069480669088112) q[1];
ry(-0.009824537714397698) q[2];
rz(1.4169008504206815) q[2];
ry(0.00024790627762340267) q[3];
rz(-1.845590099369422) q[3];
ry(1.5716285390668494) q[4];
rz(1.1491294158134693) q[4];
ry(-6.895802982143096e-05) q[5];
rz(0.48237674427514593) q[5];
ry(0.059655242862661634) q[6];
rz(-1.6994671953061398) q[6];
ry(0.6476746316817872) q[7];
rz(0.3533340021088014) q[7];
ry(-3.124216437140316) q[8];
rz(3.1237636838073075) q[8];
ry(0.011488186894774266) q[9];
rz(0.08912707154492171) q[9];
ry(6.949035571598502e-05) q[10];
rz(0.39205132449936014) q[10];
ry(3.1368911298737814) q[11];
rz(0.28094003576550874) q[11];
ry(-3.124881145863266) q[12];
rz(-0.2915377572331889) q[12];
ry(0.02930431329417987) q[13];
rz(-2.5354928670467403) q[13];
ry(-2.6837903720125293) q[14];
rz(-2.076517419687722) q[14];
ry(-2.739988125080346) q[15];
rz(0.36496852218884257) q[15];
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
ry(-0.12707227701595636) q[0];
rz(0.7083892186749163) q[0];
ry(0.585625027837409) q[1];
rz(1.410086345863811) q[1];
ry(3.0945585702421248) q[2];
rz(0.21048406079890117) q[2];
ry(-2.525653394474536) q[3];
rz(-1.2402372326133309) q[3];
ry(-0.00513071014322648) q[4];
rz(1.6511693485844292) q[4];
ry(3.141189785072532) q[5];
rz(-0.18766454085270673) q[5];
ry(1.576290556409185) q[6];
rz(-1.0364482049971482) q[6];
ry(1.9479302509418601) q[7];
rz(3.0415896923532384) q[7];
ry(-1.44321241112592) q[8];
rz(0.8098786595085444) q[8];
ry(-0.19209364033002796) q[9];
rz(-0.026386239248506284) q[9];
ry(0.0006873143745709243) q[10];
rz(2.8407613129021114) q[10];
ry(-2.826183639626246) q[11];
rz(-2.185875083753293) q[11];
ry(0.9000307294505648) q[12];
rz(3.041002577659971) q[12];
ry(0.2989868596960683) q[13];
rz(-2.749810690058186) q[13];
ry(-2.1321982611477877) q[14];
rz(-1.2545668450775125) q[14];
ry(-0.4453455111618645) q[15];
rz(0.984079702135523) q[15];
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
ry(-3.048985588673376) q[0];
rz(0.6939055611233468) q[0];
ry(-3.1324253374886304) q[1];
rz(1.982286400983965) q[1];
ry(-3.1327367181891868) q[2];
rz(0.13623745097016826) q[2];
ry(-0.0004464163706359159) q[3];
rz(-1.4364982365814365) q[3];
ry(-3.1385686791843006) q[4];
rz(2.1787332697879362) q[4];
ry(0.0015736194874025069) q[5];
rz(1.617656774516857) q[5];
ry(-0.0017590700940628423) q[6];
rz(1.4970934518630015) q[6];
ry(1.5744814315520383) q[7];
rz(1.5593574332713525) q[7];
ry(-2.524244077866002) q[8];
rz(-3.0641592243631015) q[8];
ry(3.141213438976741) q[9];
rz(0.9935677899318719) q[9];
ry(-3.1415886044449772) q[10];
rz(-0.754852564592387) q[10];
ry(1.5636230587694284) q[11];
rz(1.5495752960064983) q[11];
ry(0.32993994434921253) q[12];
rz(0.4008864025855825) q[12];
ry(-1.8967424020688817) q[13];
rz(2.507166287885752) q[13];
ry(3.0158666248196586) q[14];
rz(-2.555267709313548) q[14];
ry(1.9169171008804713) q[15];
rz(2.306148937779829) q[15];
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
ry(1.3007141414233703) q[0];
rz(1.384108456269669) q[0];
ry(1.6043648281566103) q[1];
rz(2.4318967434295153) q[1];
ry(-1.5367439148625266) q[2];
rz(-3.125118267837506) q[2];
ry(2.5317114364503523) q[3];
rz(0.8135801021517997) q[3];
ry(3.133251405584627) q[4];
rz(0.9352275183724696) q[4];
ry(-3.1361535967672416) q[5];
rz(-2.510889933878405) q[5];
ry(3.135710713375729) q[6];
rz(-2.858805589993117) q[6];
ry(-1.5491086914519077) q[7];
rz(1.6026260506409173) q[7];
ry(3.1054858672401293) q[8];
rz(-3.0219781720809693) q[8];
ry(-3.9370693428075754e-05) q[9];
rz(-2.7501870409956592) q[9];
ry(-1.5829237320157319) q[10];
rz(2.2490619467313895) q[10];
ry(-0.0339084362073041) q[11];
rz(-0.013136543489554953) q[11];
ry(3.1368663388033178) q[12];
rz(-2.530195781741118) q[12];
ry(3.1398461910957303) q[13];
rz(-0.6569819951863964) q[13];
ry(-2.901114943667608) q[14];
rz(1.1187051913088273) q[14];
ry(3.1126882650078294) q[15];
rz(1.9312252553940459) q[15];
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
ry(1.5763656399759107) q[0];
rz(1.572106975843637) q[0];
ry(-1.6843263965530602) q[1];
rz(-0.6294137981412113) q[1];
ry(-1.6967089672754536) q[2];
rz(3.1368842308933544) q[2];
ry(-0.000817388769610794) q[3];
rz(-1.9168570737860768) q[3];
ry(3.0777611192370116) q[4];
rz(-1.5343898815885681) q[4];
ry(0.0011122267939228064) q[5];
rz(3.106741298398599) q[5];
ry(0.00041111605238430826) q[6];
rz(-2.963698568923906) q[6];
ry(0.29106220940206745) q[7];
rz(1.919361277303854) q[7];
ry(1.5832893558821182) q[8];
rz(-1.5543321719877943) q[8];
ry(0.037082776077931874) q[9];
rz(0.42479806309013424) q[9];
ry(-3.141439466877692) q[10];
rz(-0.893780166534798) q[10];
ry(0.002139824626821607) q[11];
rz(1.0857277652946806) q[11];
ry(-3.1289443253329) q[12];
rz(2.635186133030321) q[12];
ry(1.2451096875660754) q[13];
rz(-1.2669137776045991) q[13];
ry(3.02609515138491) q[14];
rz(1.0261432608482117) q[14];
ry(-2.0622964792309366) q[15];
rz(1.1130571135985408) q[15];
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
ry(-1.6569039854878405) q[0];
rz(-2.8978312605257224) q[0];
ry(-0.007632783391278153) q[1];
rz(2.9563785535081437) q[1];
ry(0.6335123709254815) q[2];
rz(3.1321890492402895) q[2];
ry(-3.1354765634868604) q[3];
rz(0.3646335959528395) q[3];
ry(0.00011562168718211692) q[4];
rz(1.1910112534346846) q[4];
ry(1.5752750108054832) q[5];
rz(0.43208255867557893) q[5];
ry(1.5387815417643305) q[6];
rz(-0.9296205742217213) q[6];
ry(1.5951957043158256) q[7];
rz(0.37848370551158733) q[7];
ry(1.0795082862975462) q[8];
rz(3.0978336180548185) q[8];
ry(-3.141416143896456) q[9];
rz(0.9635739869455627) q[9];
ry(-2.1906235044186784) q[10];
rz(1.216625747235037) q[10];
ry(1.5931546834104662) q[11];
rz(1.5154878787188437) q[11];
ry(-2.4889450344488404) q[12];
rz(2.5605180420002305) q[12];
ry(-3.119789850978489) q[13];
rz(2.3304646290407978) q[13];
ry(-0.9087698882524446) q[14];
rz(-1.016151472276296) q[14];
ry(-2.712015796143735) q[15];
rz(1.9188758649943845) q[15];
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
ry(-0.03061853074163595) q[0];
rz(1.6833399486983511) q[0];
ry(-3.139686389112605) q[1];
rz(0.4251881807636963) q[1];
ry(-1.565739908069215) q[2];
rz(1.5851224271800328) q[2];
ry(1.5270384583551975) q[3];
rz(-1.5737596843760835) q[3];
ry(3.1415290927023296) q[4];
rz(3.034358603668281) q[4];
ry(0.32095511429071166) q[5];
rz(2.061787581822478) q[5];
ry(0.6573512983347509) q[6];
rz(-2.1043588824530284) q[6];
ry(-4.617557011421525e-06) q[7];
rz(-0.14649002259740396) q[7];
ry(1.5688441944301337) q[8];
rz(2.4460103443102286) q[8];
ry(3.141442629720968) q[9];
rz(-2.514718188243997) q[9];
ry(3.1408797447817167) q[10];
rz(-3.040765189690607) q[10];
ry(0.5623226943933686) q[11];
rz(-1.077275462687964) q[11];
ry(3.136734173704774) q[12];
rz(3.0637971308784784) q[12];
ry(0.011300864612926453) q[13];
rz(0.8370586535010753) q[13];
ry(-2.2810927881624194) q[14];
rz(2.1180213898531077) q[14];
ry(-1.548327477583842) q[15];
rz(-0.45790328231364086) q[15];
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
ry(-3.093729130578508) q[0];
rz(-1.21595286686369) q[0];
ry(3.1402610560410404) q[1];
rz(-2.596890511334528) q[1];
ry(3.0212487262489875) q[2];
rz(0.0476427222001794) q[2];
ry(0.025761537523492528) q[3];
rz(-1.5928660500917502) q[3];
ry(-0.0004855308929201385) q[4];
rz(1.3400913730631672) q[4];
ry(0.00017990010410190638) q[5];
rz(2.439311031784114) q[5];
ry(-1.590258266055649) q[6];
rz(3.140100495960741) q[6];
ry(-3.1384559359242044) q[7];
rz(0.23465982399256818) q[7];
ry(3.1347172757791864) q[8];
rz(-2.268251930690694) q[8];
ry(-2.1128893833299048e-06) q[9];
rz(2.201240817109663) q[9];
ry(-3.141532247785007) q[10];
rz(0.360245083755645) q[10];
ry(-2.116267747421592) q[11];
rz(-1.8220807248453212) q[11];
ry(2.565034125897069) q[12];
rz(-2.8845467392548736) q[12];
ry(1.5672565035376653) q[13];
rz(-1.4165588702385783) q[13];
ry(0.021276670348217418) q[14];
rz(-0.24265767555816353) q[14];
ry(-2.96369538539909) q[15];
rz(-1.867561471771605) q[15];
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
ry(1.5763495350564587) q[0];
rz(3.088692124601631) q[0];
ry(3.140042317910298) q[1];
rz(-2.435083133495015) q[1];
ry(-3.127903308657639) q[2];
rz(-1.8661837353180273) q[2];
ry(-1.614231715703605) q[3];
rz(1.4363426743433707) q[3];
ry(-0.00011976446880563258) q[4];
rz(0.009460420337879975) q[4];
ry(-2.155475335490287) q[5];
rz(1.3541518257236476) q[5];
ry(-1.9449814133713703) q[6];
rz(-0.013821077187381015) q[6];
ry(1.5707822861455476) q[7];
rz(3.140844824805997) q[7];
ry(-1.5757045872447009) q[8];
rz(2.681379061503943) q[8];
ry(0.00029901562482330905) q[9];
rz(-2.1885089954892036) q[9];
ry(0.0008867505012999286) q[10];
rz(1.3473618101600493) q[10];
ry(-3.1314681985256403) q[11];
rz(2.317807257076824) q[11];
ry(-2.675055431047772) q[12];
rz(-0.7285750681400422) q[12];
ry(3.1409754172664566) q[13];
rz(-2.984806447398763) q[13];
ry(1.7135003588377287) q[14];
rz(-2.725978595488819) q[14];
ry(1.5799401797964037) q[15];
rz(-0.8722895800172121) q[15];
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
ry(1.4646371288579232) q[0];
rz(1.5089706964778904) q[0];
ry(-1.569902478537727) q[1];
rz(1.1015794490840725) q[1];
ry(-3.13913673744572) q[2];
rz(-2.305126310298898) q[2];
ry(-0.0004881230556907923) q[3];
rz(0.13004687135345863) q[3];
ry(1.5482833252133883) q[4];
rz(-2.53899063260461) q[4];
ry(3.141385164448246) q[5];
rz(1.586272706869008) q[5];
ry(-1.5706722982332868) q[6];
rz(-0.03379033495331196) q[6];
ry(1.5714653372134855) q[7];
rz(-1.4015429499497722) q[7];
ry(-0.0004649345173159958) q[8];
rz(2.405141640281074) q[8];
ry(-0.00023869599008058488) q[9];
rz(3.0236777940242012) q[9];
ry(3.1400608810973845) q[10];
rz(0.38959940390985004) q[10];
ry(1.420338895955438) q[11];
rz(2.0240301128776323) q[11];
ry(1.3902683384041368) q[12];
rz(2.1882408858877946) q[12];
ry(1.5708341870274038) q[13];
rz(-1.6961049267634216) q[13];
ry(-1.5601450266533803) q[14];
rz(-1.5460712148814684) q[14];
ry(-0.00979645962030773) q[15];
rz(-2.0614154709161747) q[15];
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
ry(-0.013511346255056012) q[0];
rz(3.113682178483837) q[0];
ry(-0.14568698070650993) q[1];
rz(2.029600647163851) q[1];
ry(-1.6870891843189053) q[2];
rz(1.5009500665153126) q[2];
ry(-1.5714243685363911) q[3];
rz(3.1337853939695037) q[3];
ry(2.2277897180816306) q[4];
rz(0.038383483951486504) q[4];
ry(0.04660516562094863) q[5];
rz(-2.2833715431884296) q[5];
ry(0.005633753314303824) q[6];
rz(-1.3075866678885657) q[6];
ry(3.1352103887315956) q[7];
rz(2.5208399322094515) q[7];
ry(0.43125465390197204) q[8];
rz(-1.3937210935252695) q[8];
ry(1.673416835441056) q[9];
rz(0.01648888628301393) q[9];
ry(-4.933739168322403e-06) q[10];
rz(-0.5131702393151042) q[10];
ry(-1.5704809204762489) q[11];
rz(-2.778848282779487) q[11];
ry(-0.00254683292823632) q[12];
rz(-0.6191192361003992) q[12];
ry(0.012331959568925448) q[13];
rz(0.9977653526628721) q[13];
ry(-1.2394222127631442) q[14];
rz(1.4875639256332398) q[14];
ry(-0.13798407746447783) q[15];
rz(0.056050986938561045) q[15];
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
ry(3.12504675441202) q[0];
rz(2.7303015747612136) q[0];
ry(-2.2473520418252666) q[1];
rz(0.4221509738143432) q[1];
ry(-0.557992222503913) q[2];
rz(0.00015761884444565058) q[2];
ry(-2.4491123189918085e-05) q[3];
rz(0.1196469003796042) q[3];
ry(0.04016239993303694) q[4];
rz(0.7271268282609543) q[4];
ry(0.008695624843768596) q[5];
rz(2.274906415668654) q[5];
ry(0.0001900950444984062) q[6];
rz(1.146859171082422) q[6];
ry(-0.0007515196617754966) q[7];
rz(1.593894999349363) q[7];
ry(0.00022324837287333857) q[8];
rz(-2.113513574200043) q[8];
ry(-0.06940068119048792) q[9];
rz(3.1045199197736766) q[9];
ry(-1.5864230780595814) q[10];
rz(1.1465276946318859) q[10];
ry(-0.9457806682220724) q[11];
rz(3.059313752204014) q[11];
ry(-1.5606572786764294) q[12];
rz(1.0631301103651696) q[12];
ry(0.003966599321268305) q[13];
rz(-2.4713407268906633) q[13];
ry(-1.6007097968172097) q[14];
rz(0.16529647888358825) q[14];
ry(-1.1686768428236216) q[15];
rz(-0.15673096521068983) q[15];
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
ry(3.1414319019688492) q[0];
rz(-1.8713510467534125) q[0];
ry(-3.118164244494904) q[1];
rz(0.4589677896508544) q[1];
ry(2.0221284752091355) q[2];
rz(2.914978921753504) q[2];
ry(0.0014773813471751693) q[3];
rz(2.7325039579120514) q[3];
ry(2.010835185109121) q[4];
rz(-2.589568491888097) q[4];
ry(-0.04641002259885929) q[5];
rz(-2.07792873921444) q[5];
ry(3.1353148790827663) q[6];
rz(-1.0146599680304924) q[6];
ry(3.1160628817103038) q[7];
rz(2.3163119952412585) q[7];
ry(-1.560552882896915) q[8];
rz(1.068689405565884) q[8];
ry(-2.966261376969908) q[9];
rz(1.5532075330538218) q[9];
ry(-3.1408645326642253) q[10];
rz(-1.9990974613063859) q[10];
ry(-3.1406774759052043) q[11];
rz(0.4910274948802913) q[11];
ry(3.1410550808730675) q[12];
rz(2.6012137087651417) q[12];
ry(3.1412171428865094) q[13];
rz(-0.1538541889127386) q[13];
ry(-0.0002585385304601102) q[14];
rz(2.9170473783899245) q[14];
ry(1.5635857256722627) q[15];
rz(1.5743681783773864) q[15];
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
ry(1.5696152163029655) q[0];
rz(-0.37663883628381845) q[0];
ry(2.438933432662831) q[1];
rz(1.5985892664201793) q[1];
ry(-2.2257735736946276) q[2];
rz(0.5127150478411916) q[2];
ry(2.8663428466420777) q[3];
rz(0.04154576576602231) q[3];
ry(1.6959618139212216) q[4];
rz(0.7910112738124235) q[4];
ry(-2.4102967306867797) q[5];
rz(-0.5686812540362028) q[5];
ry(3.139877078499613) q[6];
rz(0.7908000274306076) q[6];
ry(-0.0028309154632664146) q[7];
rz(1.7326586718804888) q[7];
ry(-0.0006588881831790516) q[8];
rz(0.8545693167771349) q[8];
ry(0.00358422473459008) q[9];
rz(0.4884321281536081) q[9];
ry(2.1490106725048546) q[10];
rz(2.2779438976819186) q[10];
ry(2.424378684263149) q[11];
rz(-2.8238382363680916) q[11];
ry(0.16544506994337024) q[12];
rz(1.6210466570365916) q[12];
ry(-0.014635516611521737) q[13];
rz(0.047189900304181784) q[13];
ry(-0.14878421144530066) q[14];
rz(1.4342028597593026) q[14];
ry(1.5771159191756903) q[15];
rz(-2.7299588263806793) q[15];
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
ry(3.130120696472566) q[0];
rz(-0.3964365003704433) q[0];
ry(-1.8146053988908883) q[1];
rz(-1.4580116269400731) q[1];
ry(-3.1405875983884326) q[2];
rz(-2.8897726091717613) q[2];
ry(3.141105972482667) q[3];
rz(1.8599761448321057) q[3];
ry(0.056108269841357306) q[4];
rz(0.6027787847817947) q[4];
ry(-0.00023632826435004972) q[5];
rz(-2.572880863128988) q[5];
ry(0.08766225094624756) q[6];
rz(-0.9936060719733533) q[6];
ry(0.0003917619555673548) q[7];
rz(1.2624021603701807) q[7];
ry(-3.0885213089864934) q[8];
rz(-2.7575731465840136) q[8];
ry(3.096581510250515) q[9];
rz(0.5337031845550779) q[9];
ry(-3.1285887965633448) q[10];
rz(0.6988081687766784) q[10];
ry(0.0005220540064516721) q[11];
rz(0.011473299377346535) q[11];
ry(-1.5723724433970743) q[12];
rz(2.0815025285118622) q[12];
ry(1.570925404803627) q[13];
rz(1.6136835312273548) q[13];
ry(1.537411086727921) q[14];
rz(0.008299809621971297) q[14];
ry(1.4105096874143876) q[15];
rz(1.7010968183425588) q[15];
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
ry(0.0024222447657648487) q[0];
rz(-3.121728056940528) q[0];
ry(-0.0013407181908933197) q[1];
rz(3.0282000804135727) q[1];
ry(1.271317295624324) q[2];
rz(-2.336356486891175) q[2];
ry(-0.31884892112140306) q[3];
rz(-0.21245700980653834) q[3];
ry(-2.9832865421235875) q[4];
rz(-0.1901093615782452) q[4];
ry(2.41674075376638) q[5];
rz(0.5591183639486248) q[5];
ry(3.139920335482926) q[6];
rz(-2.5234127066852454) q[6];
ry(-3.1379075358328397) q[7];
rz(1.3646156736486912) q[7];
ry(-0.0006642061920807019) q[8];
rz(0.1499606260271623) q[8];
ry(0.004027475442928363) q[9];
rz(-0.5448468121045416) q[9];
ry(0.31309578666244764) q[10];
rz(0.419582217261624) q[10];
ry(-3.137864033516831) q[11];
rz(2.5569362591887494) q[11];
ry(-3.141379038570317) q[12];
rz(-1.0598623957591888) q[12];
ry(3.1201711725271433) q[13];
rz(1.613361223040032) q[13];
ry(-1.5711345765560945) q[14];
rz(1.5625644754060417) q[14];
ry(1.5706732365199367) q[15];
rz(-0.21397108304968704) q[15];
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
ry(-2.5632018083013097) q[0];
rz(0.6072837328013245) q[0];
ry(-1.5714700881896393) q[1];
rz(2.4911664588236224) q[1];
ry(-3.140460898139365) q[2];
rz(1.2274540564745244) q[2];
ry(0.00037871077836550685) q[3];
rz(2.9884656671602094) q[3];
ry(-1.5645676386229823) q[4];
rz(-0.7896387324531859) q[4];
ry(3.141307280899424) q[5];
rz(1.7950263581908414) q[5];
ry(-1.57461710136249) q[6];
rz(2.0628790283023575) q[6];
ry(-1.5822209193819832) q[7];
rz(2.713560497994039) q[7];
ry(3.1404004406055908) q[8];
rz(-0.7842429354529761) q[8];
ry(-3.093623224217953) q[9];
rz(-2.4734222593980935) q[9];
ry(-0.011938808422993018) q[10];
rz(-1.3741572494505832) q[10];
ry(-3.141261879652316) q[11];
rz(2.2804841726476033) q[11];
ry(1.5716928130031718) q[12];
rz(2.1812793157836463) q[12];
ry(-1.5758537334162925) q[13];
rz(-0.39970331147069066) q[13];
ry(-0.1290337542165829) q[14];
rz(0.6177882884970307) q[14];
ry(-3.141494346066436) q[15];
rz(-0.6129000555651084) q[15];