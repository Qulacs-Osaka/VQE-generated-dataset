OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
ry(-0.6554861112453034) q[0];
rz(0.6252607974646607) q[0];
ry(0.7618781392640173) q[1];
rz(2.1094422552964374) q[1];
ry(0.3751739694425072) q[2];
rz(1.775955560373059) q[2];
ry(1.272527076553878) q[3];
rz(1.8585391014578967) q[3];
ry(-0.5614771515060832) q[4];
rz(-2.085430719339535) q[4];
ry(-1.1468806709484252) q[5];
rz(3.12160224473937) q[5];
ry(0.2558021877902981) q[6];
rz(-0.492505779659811) q[6];
ry(1.3655339619111664) q[7];
rz(0.8965735831889106) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.5146362470169157) q[0];
rz(2.462418039975149) q[0];
ry(1.583937907498079) q[1];
rz(-0.3002070426929225) q[1];
ry(2.8364281950580374) q[2];
rz(-0.7995187573783029) q[2];
ry(2.4040562360186084) q[3];
rz(0.14125550823812194) q[3];
ry(2.1339301509164725) q[4];
rz(-1.615586607948267) q[4];
ry(-1.7628686245302383) q[5];
rz(-1.9226901816542263) q[5];
ry(-1.6975778077068364) q[6];
rz(2.6051092058956544) q[6];
ry(2.667751142455117) q[7];
rz(-1.8034544802968695) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.9772106938066463) q[0];
rz(2.483295557029899) q[0];
ry(2.752804729489014) q[1];
rz(-0.37165272929115534) q[1];
ry(1.4621602995118321) q[2];
rz(-0.8677310493256595) q[2];
ry(1.0200706127244952) q[3];
rz(-1.324167637430155) q[3];
ry(-1.079646625654048) q[4];
rz(1.7728956071282447) q[4];
ry(2.2129928996868093) q[5];
rz(-1.5887787291576787) q[5];
ry(-0.4963437150008886) q[6];
rz(-2.9956667356700444) q[6];
ry(0.16614011286553138) q[7];
rz(2.787583481261624) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(0.5839313851501906) q[0];
rz(2.759600149639705) q[0];
ry(-2.657932733304519) q[1];
rz(0.4891539903689773) q[1];
ry(1.5206430146034222) q[2];
rz(-0.386417289725363) q[2];
ry(-1.7208415386441107) q[3];
rz(-2.3298028347512147) q[3];
ry(3.0468185669180463) q[4];
rz(-1.3707563553293935) q[4];
ry(1.6825907069111752) q[5];
rz(0.3481455382096739) q[5];
ry(1.7609125288514615) q[6];
rz(0.39174892433644093) q[6];
ry(-1.2014974931331346) q[7];
rz(2.279254052981903) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-0.45766457744608896) q[0];
rz(2.377829007816574) q[0];
ry(1.8421352040003045) q[1];
rz(-0.11739825470476804) q[1];
ry(1.0962913120748157) q[2];
rz(-1.9901421377782935) q[2];
ry(0.809999466196473) q[3];
rz(-1.475257492131634) q[3];
ry(2.3931454128030785) q[4];
rz(1.968511429346902) q[4];
ry(-1.6246330490282943) q[5];
rz(2.0931348080135357) q[5];
ry(0.8066344165122329) q[6];
rz(-2.6692858640424864) q[6];
ry(-2.8385345537969404) q[7];
rz(-3.0227936447881056) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-0.3546731333291486) q[0];
rz(0.6974142332400249) q[0];
ry(0.20318411843818393) q[1];
rz(-1.8987055084628546) q[1];
ry(-2.5700682182372385) q[2];
rz(-0.8424363249161645) q[2];
ry(2.23858897078615) q[3];
rz(-1.3222691422483113) q[3];
ry(-2.917852714702214) q[4];
rz(0.9766831748862186) q[4];
ry(-0.5151259420970469) q[5];
rz(-2.9189029375981708) q[5];
ry(0.7806069821035215) q[6];
rz(-1.5195097453517767) q[6];
ry(-2.7108404018203753) q[7];
rz(2.7174478349291515) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-0.7697642935383973) q[0];
rz(-1.0586111949769235) q[0];
ry(-0.7038101185372067) q[1];
rz(2.6440694491804333) q[1];
ry(2.155611894895224) q[2];
rz(1.6636636417238726) q[2];
ry(1.0916427429043958) q[3];
rz(-0.8036573618271909) q[3];
ry(2.578320280640084) q[4];
rz(-3.016850801035335) q[4];
ry(2.769536130324752) q[5];
rz(-1.7658463326182845) q[5];
ry(1.725070330955257) q[6];
rz(1.6541085348787234) q[6];
ry(0.2879531694145193) q[7];
rz(1.1535947008692213) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-0.3154355180830759) q[0];
rz(2.5637723670237325) q[0];
ry(1.6675023749253768) q[1];
rz(1.0076183682826902) q[1];
ry(1.828069412756505) q[2];
rz(-0.735176482706028) q[2];
ry(1.6758436289318048) q[3];
rz(2.064194085628532) q[3];
ry(1.2216831255631595) q[4];
rz(-0.1423777016466117) q[4];
ry(-0.2232184837696272) q[5];
rz(2.321273197916027) q[5];
ry(-1.7895322712419055) q[6];
rz(-1.2388041890582489) q[6];
ry(-0.4175144112968683) q[7];
rz(0.8204604556732882) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(1.631029099543059) q[0];
rz(2.0219147991152293) q[0];
ry(1.2234400599828668) q[1];
rz(-0.9916324867258308) q[1];
ry(2.0187416831791554) q[2];
rz(0.23502627283371783) q[2];
ry(-2.3533727309625894) q[3];
rz(2.862134470595296) q[3];
ry(1.3642925101179273) q[4];
rz(1.035480687873376) q[4];
ry(-1.822007853872467) q[5];
rz(-0.42236687904303327) q[5];
ry(-2.3534388822896237) q[6];
rz(-2.0309091631091767) q[6];
ry(-2.7315465035417836) q[7];
rz(0.29398075775879523) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-0.2321996172177272) q[0];
rz(0.242379446462599) q[0];
ry(2.484164337114868) q[1];
rz(0.906500548522998) q[1];
ry(-1.6273740694837144) q[2];
rz(0.5969913558614905) q[2];
ry(2.569993618330454) q[3];
rz(-1.886247222761798) q[3];
ry(0.14960809742757086) q[4];
rz(-0.14995026000758305) q[4];
ry(1.222376524668045) q[5];
rz(-0.14624733319482355) q[5];
ry(-2.2639894656370196) q[6];
rz(-1.4347814848959142) q[6];
ry(1.0453046184167558) q[7];
rz(-0.07425881964164929) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(1.7750366693704809) q[0];
rz(-2.5696110075051277) q[0];
ry(-1.164591405356659) q[1];
rz(1.9015780300589002) q[1];
ry(-2.24483936278009) q[2];
rz(2.945319954578358) q[2];
ry(1.4473210613851297) q[3];
rz(-1.4838192522717761) q[3];
ry(-2.536916271279267) q[4];
rz(1.6402524214910887) q[4];
ry(1.655659577878364) q[5];
rz(2.989474669004542) q[5];
ry(-2.5012381869741422) q[6];
rz(-0.7227969333152547) q[6];
ry(-2.0658152302712756) q[7];
rz(-2.679227345069316) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-0.32032138494912715) q[0];
rz(-0.26508679325577145) q[0];
ry(1.8502615688694926) q[1];
rz(-0.3639665036113851) q[1];
ry(2.7626151896079567) q[2];
rz(-0.6694168562863253) q[2];
ry(0.21715884115156855) q[3];
rz(-2.6449707501324453) q[3];
ry(0.9743146446438012) q[4];
rz(-3.1036556658067087) q[4];
ry(-2.250100016256871) q[5];
rz(-2.212803393477999) q[5];
ry(-0.7305333677201468) q[6];
rz(-2.114819038686627) q[6];
ry(-1.2166802735733313) q[7];
rz(-1.8975581429779735) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.4455926176035714) q[0];
rz(0.9052892249111689) q[0];
ry(2.7514797083046725) q[1];
rz(2.1997040888971515) q[1];
ry(2.198931783112015) q[2];
rz(-0.477298067704102) q[2];
ry(1.3258847372236902) q[3];
rz(1.5451840729425153) q[3];
ry(-1.5450454856823352) q[4];
rz(-0.057549181576370685) q[4];
ry(1.3455649102340914) q[5];
rz(-1.1931431241127175) q[5];
ry(1.0557289184808083) q[6];
rz(-2.070686062969692) q[6];
ry(2.8323827684950666) q[7];
rz(0.6365310497886696) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.5958734694035153) q[0];
rz(-0.6661362583511625) q[0];
ry(1.016425107679666) q[1];
rz(2.5136719514069363) q[1];
ry(1.935432592172111) q[2];
rz(-0.304513932495514) q[2];
ry(0.40198796860082897) q[3];
rz(-1.6117294190661697) q[3];
ry(0.7504602493022032) q[4];
rz(-2.6962416232337043) q[4];
ry(-2.602493109392309) q[5];
rz(1.8099737973277277) q[5];
ry(-1.0370397273218523) q[6];
rz(-0.03229504938826722) q[6];
ry(-0.82754327905607) q[7];
rz(1.8206234452443892) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.3409270286919797) q[0];
rz(-1.816849283785892) q[0];
ry(1.68659892536699) q[1];
rz(-1.0222766096678715) q[1];
ry(-2.7208459777090255) q[2];
rz(0.4303368351167279) q[2];
ry(0.752446040440501) q[3];
rz(-0.6791758479088709) q[3];
ry(-2.3536912331016375) q[4];
rz(0.7426637715871998) q[4];
ry(-1.8140507469093121) q[5];
rz(2.699500828143126) q[5];
ry(-2.011609942506631) q[6];
rz(1.5498166438926069) q[6];
ry(0.18771819493542874) q[7];
rz(-2.0967315781897002) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-0.45087641466655) q[0];
rz(0.6616799520581367) q[0];
ry(0.8868050765908483) q[1];
rz(-2.6975277656927292) q[1];
ry(-2.7069738999025628) q[2];
rz(0.6004515373684421) q[2];
ry(-1.2767886124692085) q[3];
rz(0.5124706238291186) q[3];
ry(-2.016770195112538) q[4];
rz(-1.904738859903297) q[4];
ry(-0.9787422605349736) q[5];
rz(-0.34221534635140843) q[5];
ry(2.4146645182913806) q[6];
rz(-0.02433812225739616) q[6];
ry(-1.1026358091637976) q[7];
rz(2.843303206217892) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(2.3883534125779664) q[0];
rz(-1.6147843817742036) q[0];
ry(1.1273970346475846) q[1];
rz(0.9485248797878104) q[1];
ry(2.5825942490274807) q[2];
rz(-1.200294163777927) q[2];
ry(2.8364856388590227) q[3];
rz(-0.35145334449377597) q[3];
ry(-0.8160294650316375) q[4];
rz(-2.1082425549447836) q[4];
ry(1.4585932165919413) q[5];
rz(-1.9616888392730925) q[5];
ry(-0.8688913916118457) q[6];
rz(-2.906449270264876) q[6];
ry(-1.0990644218289036) q[7];
rz(3.1072072793376577) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(1.7005930840175134) q[0];
rz(1.7824910032419075) q[0];
ry(1.8004877549553084) q[1];
rz(0.5340539644978723) q[1];
ry(2.3872039372752183) q[2];
rz(-0.24189840054712194) q[2];
ry(-2.67871541851397) q[3];
rz(2.755041608816629) q[3];
ry(1.231915843303522) q[4];
rz(-1.6295526720655458) q[4];
ry(2.778488639343732) q[5];
rz(0.98155517948204) q[5];
ry(0.39440959658773256) q[6];
rz(1.6637519522896616) q[6];
ry(0.3885638047552096) q[7];
rz(-0.0021057258381338424) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-0.5046058650823334) q[0];
rz(-0.7072485482871027) q[0];
ry(0.2479912078414347) q[1];
rz(-3.115955356779602) q[1];
ry(-1.147297165355093) q[2];
rz(-3.103094965504563) q[2];
ry(0.7043204734067243) q[3];
rz(0.13287009614311085) q[3];
ry(2.346683523051317) q[4];
rz(1.124694399808095) q[4];
ry(-2.036373739805564) q[5];
rz(1.813657762096055) q[5];
ry(-1.5665879277661343) q[6];
rz(-2.9222014060837873) q[6];
ry(1.1826913088258282) q[7];
rz(-0.13470781723311198) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(0.9902666667571017) q[0];
rz(-1.0017830612192222) q[0];
ry(-0.5825171108590824) q[1];
rz(1.4655566931207078) q[1];
ry(1.7044683887108614) q[2];
rz(-0.8517539307552378) q[2];
ry(1.6888315829543497) q[3];
rz(-2.6319741781634023) q[3];
ry(-1.7558683426555914) q[4];
rz(-3.0604144155231245) q[4];
ry(2.252357869239509) q[5];
rz(-2.790729574134577) q[5];
ry(-2.6052134086939245) q[6];
rz(-2.429062929427998) q[6];
ry(-0.9555808864249095) q[7];
rz(-2.8180621120419156) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(1.3059598894252065) q[0];
rz(2.010654215318242) q[0];
ry(-0.18052292321646923) q[1];
rz(1.8326070939972068) q[1];
ry(-1.267762762396785) q[2];
rz(0.9923189366313964) q[2];
ry(2.1327417944399913) q[3];
rz(-2.0165873541560613) q[3];
ry(-0.7284799572721059) q[4];
rz(1.7074270023558547) q[4];
ry(1.8541042306292357) q[5];
rz(2.66410353803135) q[5];
ry(-1.2497451083681959) q[6];
rz(-1.9198755878647598) q[6];
ry(1.0420175302763381) q[7];
rz(-2.4583838865602248) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-2.3317426103020362) q[0];
rz(-1.5603387502781247) q[0];
ry(0.7033360026323345) q[1];
rz(-2.430253335999491) q[1];
ry(1.2867495780159397) q[2];
rz(2.6899189078065953) q[2];
ry(0.611192554496566) q[3];
rz(1.9822453454644746) q[3];
ry(0.7217080421193666) q[4];
rz(-0.952028623081679) q[4];
ry(-2.366778462821931) q[5];
rz(-0.8172440321545783) q[5];
ry(0.4235109871720007) q[6];
rz(0.9564848749524106) q[6];
ry(-0.2917200866921948) q[7];
rz(-0.44481748514444996) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.9696923952676055) q[0];
rz(3.036996912039337) q[0];
ry(-2.3728886548278436) q[1];
rz(-0.2573955077912508) q[1];
ry(-2.3737981016515444) q[2];
rz(0.7628079261608574) q[2];
ry(1.7997368230035715) q[3];
rz(-0.6382220968750023) q[3];
ry(1.5423536386054115) q[4];
rz(-1.9118243312009302) q[4];
ry(1.3126795285733044) q[5];
rz(-1.9993112573102592) q[5];
ry(-2.6420614327716008) q[6];
rz(3.0389103110665214) q[6];
ry(-0.4229651467623504) q[7];
rz(1.9526062729053688) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(2.6376899565818466) q[0];
rz(1.1262213743544978) q[0];
ry(1.5775840180647203) q[1];
rz(-0.6805966893685627) q[1];
ry(2.779807469946664) q[2];
rz(-0.09477013628010955) q[2];
ry(-0.298777473508767) q[3];
rz(-3.018630497123123) q[3];
ry(-0.8797326702137006) q[4];
rz(-2.41965233100317) q[4];
ry(-1.338359999533541) q[5];
rz(1.7263383490370376) q[5];
ry(-1.589062376463779) q[6];
rz(-2.63540351492844) q[6];
ry(-1.871542523727414) q[7];
rz(2.131214750823443) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(0.2374476211282624) q[0];
rz(0.7192630399493107) q[0];
ry(0.9574241793539262) q[1];
rz(3.0417873019605217) q[1];
ry(1.9819645978550442) q[2];
rz(0.7768292797647062) q[2];
ry(0.9640813580521624) q[3];
rz(0.4961028233768374) q[3];
ry(1.726004973845362) q[4];
rz(-2.8670021715899434) q[4];
ry(1.3231162741416405) q[5];
rz(2.805556979212503) q[5];
ry(1.240872084988899) q[6];
rz(-2.4633896578935093) q[6];
ry(-2.856027909615644) q[7];
rz(-2.1291868991994227) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.417180645344665) q[0];
rz(2.0348557538155276) q[0];
ry(-1.1015048500510884) q[1];
rz(-0.9158820909938603) q[1];
ry(-1.9817787556161142) q[2];
rz(-1.9519532965808337) q[2];
ry(0.7812479175949384) q[3];
rz(3.096492608693421) q[3];
ry(-0.9875152420035005) q[4];
rz(-1.7199931411445073) q[4];
ry(-1.951722413696584) q[5];
rz(2.6538283015124056) q[5];
ry(-0.7847637010764315) q[6];
rz(0.5228781972955954) q[6];
ry(-2.292750079507831) q[7];
rz(-1.9016733968116881) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-0.6617308106676054) q[0];
rz(2.2546699769643803) q[0];
ry(2.3148405578505096) q[1];
rz(-1.1634607691618184) q[1];
ry(-0.3273005972139912) q[2];
rz(-2.103033484668145) q[2];
ry(1.073471197933027) q[3];
rz(-2.985551760989516) q[3];
ry(-0.9791900026455784) q[4];
rz(-1.5369108606577386) q[4];
ry(1.5707716552942879) q[5];
rz(2.282370801518472) q[5];
ry(0.2971030305937017) q[6];
rz(-2.3284810614377642) q[6];
ry(0.7606185494386037) q[7];
rz(-1.8728157218507828) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[0],q[2];
cz q[2],q[4];
cz q[4],q[6];
cz q[1],q[3];
cz q[3],q[5];
cz q[5],q[7];
ry(-1.1452117150677534) q[0];
rz(-1.5789353930612566) q[0];
ry(-2.514981066616457) q[1];
rz(-1.551561104620843) q[1];
ry(0.32457732893965063) q[2];
rz(2.863474440215183) q[2];
ry(-1.575431593399827) q[3];
rz(-2.0809129096627608) q[3];
ry(-1.7667196421407079) q[4];
rz(-2.9547153081051305) q[4];
ry(-1.9608142439949285) q[5];
rz(3.0694336630723202) q[5];
ry(-2.106535772508611) q[6];
rz(-3.0796952372143336) q[6];
ry(1.1343810146339033) q[7];
rz(-2.6726391881653764) q[7];