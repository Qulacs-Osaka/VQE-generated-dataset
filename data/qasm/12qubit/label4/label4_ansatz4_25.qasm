OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(1.2188136418895148) q[0];
rz(-1.67265059838915) q[0];
ry(1.247525430833354) q[1];
rz(3.105590016933332) q[1];
ry(1.7303225964002005) q[2];
rz(-2.4695148019018753) q[2];
ry(-0.6636813488123929) q[3];
rz(1.547484271778008) q[3];
ry(-1.1087034097777249) q[4];
rz(-0.23801931115683317) q[4];
ry(-1.3621663704285174) q[5];
rz(0.48717629570192766) q[5];
ry(1.6054458272325505) q[6];
rz(1.6929789090371665) q[6];
ry(1.5004377852933803) q[7];
rz(1.165133591220689) q[7];
ry(-1.330126509839419) q[8];
rz(0.6139811326612623) q[8];
ry(1.6315266074977968) q[9];
rz(2.9993039370731815) q[9];
ry(-2.2862309569670822) q[10];
rz(-0.9203532169119732) q[10];
ry(-1.4199855439131683) q[11];
rz(2.843106968976105) q[11];
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
ry(-2.4281384711887073) q[0];
rz(0.9731920520921891) q[0];
ry(-3.0528921034878733) q[1];
rz(3.067565263847063) q[1];
ry(2.7845999574368214) q[2];
rz(-0.27481876541588335) q[2];
ry(-1.3742595963646658) q[3];
rz(0.14389431882291898) q[3];
ry(-0.4466713507446416) q[4];
rz(2.5184039157063594) q[4];
ry(1.319657123578713) q[5];
rz(1.5766241653224364) q[5];
ry(-0.006088967553441478) q[6];
rz(3.0212988997791608) q[6];
ry(0.002050942789527887) q[7];
rz(-1.1515366462127927) q[7];
ry(-0.0018610963103879996) q[8];
rz(-0.690389004943871) q[8];
ry(0.001952483518717917) q[9];
rz(-2.996617439283668) q[9];
ry(-0.30359570265223845) q[10];
rz(-1.8432487681057692) q[10];
ry(2.910972903448833) q[11];
rz(-2.5454928959806344) q[11];
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
ry(-1.291533888132698) q[0];
rz(-0.2668899842148699) q[0];
ry(2.388102123600059) q[1];
rz(2.953448987900477) q[1];
ry(0.23925942286706103) q[2];
rz(-1.258622920012769) q[2];
ry(3.0240156977395336) q[3];
rz(-2.6367950741329667) q[3];
ry(0.5914515237779301) q[4];
rz(1.8970327172290589) q[4];
ry(-0.7784596213305384) q[5];
rz(-1.6041066626077152) q[5];
ry(1.488793173190352) q[6];
rz(-2.7869716511496967) q[6];
ry(-0.018912134354960804) q[7];
rz(-1.7287395603918936) q[7];
ry(-1.1370066051649153) q[8];
rz(-0.3802321444371172) q[8];
ry(-1.4976777815535305) q[9];
rz(0.42985539001284767) q[9];
ry(-1.8414075346736662) q[10];
rz(-2.564237086387007) q[10];
ry(3.107467456567379) q[11];
rz(-0.5104708891336927) q[11];
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
ry(2.5943805281215497) q[0];
rz(-2.8374748494844932) q[0];
ry(-0.5626010164472474) q[1];
rz(-1.394483067060249) q[1];
ry(-2.8441306198730887) q[2];
rz(1.2451455986559974) q[2];
ry(-2.0544809719787693) q[3];
rz(-0.23429583880215302) q[3];
ry(-1.7336596211144624) q[4];
rz(2.476362987750586) q[4];
ry(2.401531568388874) q[5];
rz(2.9044138256651677) q[5];
ry(-0.027417444176162282) q[6];
rz(2.8149700246535825) q[6];
ry(3.1282912985832434) q[7];
rz(-2.0296787575216095) q[7];
ry(-3.141492803158655) q[8];
rz(2.064310684184969) q[8];
ry(-3.140753510206599) q[9];
rz(1.0503374513015449) q[9];
ry(-2.929200719124369) q[10];
rz(-2.909850198106578) q[10];
ry(2.430358170655994) q[11];
rz(0.3348467243278623) q[11];
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
ry(-2.4403028923066485) q[0];
rz(1.9598321933011285) q[0];
ry(-3.078938948729109) q[1];
rz(-0.7644123787264189) q[1];
ry(-3.113776605932699) q[2];
rz(-1.36183106202298) q[2];
ry(-2.779141354064631) q[3];
rz(-1.1442727969190842) q[3];
ry(-1.5064641307255697) q[4];
rz(0.6198840513850659) q[4];
ry(-1.456973790383019) q[5];
rz(-2.5382997955470024) q[5];
ry(-2.418408132839054) q[6];
rz(-2.796662339613508) q[6];
ry(-0.024305541056927643) q[7];
rz(0.3579753961732521) q[7];
ry(-1.5317607524235761) q[8];
rz(-0.5118879560810462) q[8];
ry(1.5887193919789815) q[9];
rz(0.4239298114565396) q[9];
ry(-1.5196398155048783) q[10];
rz(0.26127717586997706) q[10];
ry(1.7658683453239044) q[11];
rz(1.1108912746373856) q[11];
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
ry(-0.5036544615285896) q[0];
rz(-2.937408995531582) q[0];
ry(-1.6737584343806557) q[1];
rz(-0.9335898058098131) q[1];
ry(1.4089071152684125) q[2];
rz(-2.2475903901390986) q[2];
ry(-1.5121268046672807) q[3];
rz(0.9699375855763401) q[3];
ry(1.5308632775162527) q[4];
rz(0.6714650582850776) q[4];
ry(-0.47540839010421226) q[5];
rz(-2.7855007006241896) q[5];
ry(1.583032507554812) q[6];
rz(-1.5641075098931294) q[6];
ry(-1.5712980719229428) q[7];
rz(1.652780847419861) q[7];
ry(3.1407287178824737) q[8];
rz(2.255264583659371) q[8];
ry(0.0025045410341162366) q[9];
rz(-0.8287299220912976) q[9];
ry(-2.9514925385169346) q[10];
rz(0.47448610146408576) q[10];
ry(0.19237982072429283) q[11];
rz(-1.7908548242063078) q[11];
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
ry(-2.729177183450049) q[0];
rz(0.6479656006543619) q[0];
ry(2.1302074593244043) q[1];
rz(3.0062678743289712) q[1];
ry(2.5434996074501623) q[2];
rz(-3.1203702531524575) q[2];
ry(-0.6122275195459244) q[3];
rz(1.0496166855650397) q[3];
ry(3.1276537246676055) q[4];
rz(-0.29873762629174166) q[4];
ry(3.1119734630954285) q[5];
rz(-2.756923410306203) q[5];
ry(-1.1496174399946448) q[6];
rz(3.12825128367665) q[6];
ry(-3.125381106729083) q[7];
rz(-3.054337977764254) q[7];
ry(-3.141120299788875) q[8];
rz(-3.0257746210377543) q[8];
ry(-0.0018579358001134983) q[9];
rz(1.8603683699436147) q[9];
ry(-2.2441246532640395) q[10];
rz(1.3048203423551028) q[10];
ry(3.1233602623602823) q[11];
rz(1.550134792599553) q[11];
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
ry(1.8149169730038048) q[0];
rz(2.48413940341401) q[0];
ry(-1.7627101041039104) q[1];
rz(-1.5497185195402523) q[1];
ry(2.077658843209133) q[2];
rz(1.1057372871344318) q[2];
ry(-2.7084561586065896) q[3];
rz(2.5475676781823973) q[3];
ry(2.7818613032725197) q[4];
rz(2.645679388935486) q[4];
ry(2.8057069254197953) q[5];
rz(-1.8657770571023695) q[5];
ry(2.768093580932057) q[6];
rz(-1.4149170091809278) q[6];
ry(-1.575445112266391) q[7];
rz(-2.40266875472415) q[7];
ry(0.0718786915364582) q[8];
rz(0.6072257727953527) q[8];
ry(0.2729420958960684) q[9];
rz(1.5524157218815606) q[9];
ry(1.0754435906010524) q[10];
rz(-1.783601423965532) q[10];
ry(-2.9734067559697217) q[11];
rz(-0.8743645170292057) q[11];
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
ry(-2.506718980595638) q[0];
rz(-1.5641155536209483) q[0];
ry(-2.616244150177324) q[1];
rz(-2.0127412568710383) q[1];
ry(1.9908965820127442) q[2];
rz(-2.4536767110766817) q[2];
ry(2.1762702034711308) q[3];
rz(0.6811698578844021) q[3];
ry(-0.09002654192152358) q[4];
rz(-0.46135261144620626) q[4];
ry(2.6948957409593675) q[5];
rz(1.3957798482939676) q[5];
ry(-3.1314160605183687) q[6];
rz(-3.0124955926568093) q[6];
ry(0.026865936301408766) q[7];
rz(-2.2723410024317086) q[7];
ry(-3.0879916998697343) q[8];
rz(-0.4306099950773621) q[8];
ry(-2.6399038641909054) q[9];
rz(-1.7289576686530603) q[9];
ry(-1.5612088497648888) q[10];
rz(-2.9031837009358066) q[10];
ry(-0.6211446084747767) q[11];
rz(1.1626567574307032) q[11];
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
ry(0.7221149919734643) q[0];
rz(-2.9088213521013278) q[0];
ry(-2.5027231096562033) q[1];
rz(2.535443686540416) q[1];
ry(-1.0297067917557703) q[2];
rz(0.21139946191916262) q[2];
ry(1.912458463467397) q[3];
rz(-0.5956444315869153) q[3];
ry(-2.8862192244369576) q[4];
rz(1.7748498361925211) q[4];
ry(-0.6911687677963751) q[5];
rz(-2.240486886177555) q[5];
ry(1.5702097986087946) q[6];
rz(-1.3466153664485272) q[6];
ry(1.5744899465466586) q[7];
rz(-1.0732482332278295) q[7];
ry(1.5580035615501169) q[8];
rz(-2.921882716406562) q[8];
ry(1.4204667582752697) q[9];
rz(1.480614695972288) q[9];
ry(1.5066460399246866) q[10];
rz(2.3852119475495317) q[10];
ry(0.008452298835536885) q[11];
rz(2.009507381199609) q[11];
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
ry(-2.1621879002137456) q[0];
rz(1.1490947136208403) q[0];
ry(1.920634690787323) q[1];
rz(3.092212781347622) q[1];
ry(-1.2504925224025234) q[2];
rz(0.5841681665692339) q[2];
ry(0.9087332513856969) q[3];
rz(1.740937866544316) q[3];
ry(2.884979565225572) q[4];
rz(-0.8327734665609405) q[4];
ry(-3.0200028986056253) q[5];
rz(1.9291007626594159) q[5];
ry(0.003908364485914362) q[6];
rz(-1.189203652177163) q[6];
ry(3.138772415848048) q[7];
rz(1.592888502716943) q[7];
ry(-1.9260874041524279) q[8];
rz(-0.008287960582092892) q[8];
ry(1.5181013003481791) q[9];
rz(-0.3749439928930817) q[9];
ry(-3.0935151249944934) q[10];
rz(-1.294519207103141) q[10];
ry(-1.618645001373868) q[11];
rz(2.4327580436643124) q[11];
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
ry(-0.9052363124617544) q[0];
rz(3.049816596999635) q[0];
ry(-0.7907018887540982) q[1];
rz(-3.0216684203860558) q[1];
ry(-2.911189832947586) q[2];
rz(1.727470818365271) q[2];
ry(-1.5189126462953522) q[3];
rz(-2.4683938648971955) q[3];
ry(-0.2987336481680689) q[4];
rz(0.8487639202152621) q[4];
ry(-2.9149310476339765) q[5];
rz(-2.1329951299596956) q[5];
ry(0.0026723664452259536) q[6];
rz(0.9788906607294106) q[6];
ry(3.1398136632855005) q[7];
rz(-0.05005902778738047) q[7];
ry(2.6603858231830655) q[8];
rz(2.527230615635678) q[8];
ry(3.018522205282312) q[9];
rz(1.7174292811693759) q[9];
ry(2.115386012564966) q[10];
rz(2.0752386716931097) q[10];
ry(-3.0986694258175103) q[11];
rz(-1.8070243916254212) q[11];
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
ry(1.6396666142288927) q[0];
rz(1.2691448336322333) q[0];
ry(2.1831748055348745) q[1];
rz(-2.737497011179001) q[1];
ry(0.6104821316569863) q[2];
rz(-2.2498696746425555) q[2];
ry(2.437365997381926) q[3];
rz(2.0793118163718165) q[3];
ry(2.714086599353103) q[4];
rz(0.3252402262021232) q[4];
ry(-1.7343953246261181) q[5];
rz(-0.10471879565487552) q[5];
ry(0.2310812634705357) q[6];
rz(0.7446351298847135) q[6];
ry(-3.1396996785197997) q[7];
rz(-3.108913538597284) q[7];
ry(-0.8379125321887931) q[8];
rz(0.6705039497988174) q[8];
ry(-0.08833478260687766) q[9];
rz(-1.9403528429092856) q[9];
ry(-3.0983063808185145) q[10];
rz(2.465678970940341) q[10];
ry(-0.006628069978130284) q[11];
rz(-2.028221449400272) q[11];
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
ry(2.405441449879459) q[0];
rz(2.01157388391368) q[0];
ry(-2.812271055677069) q[1];
rz(-2.4393307067317713) q[1];
ry(2.9274861124232534) q[2];
rz(-0.9136122753106553) q[2];
ry(1.9719295804475312) q[3];
rz(-1.3706211799612902) q[3];
ry(1.5269294046180804) q[4];
rz(-1.6411801746922947) q[4];
ry(1.5352947960589527) q[5];
rz(1.7697093308810188) q[5];
ry(-3.1362600382503816) q[6];
rz(-1.7320636324388619) q[6];
ry(-3.1325432224005865) q[7];
rz(-0.6507906311646086) q[7];
ry(1.2324193812519724) q[8];
rz(-0.007529909579444548) q[8];
ry(0.02183584334238927) q[9];
rz(-0.802951212207772) q[9];
ry(-0.8015486315396602) q[10];
rz(-0.24858011834385607) q[10];
ry(-0.02827060490594683) q[11];
rz(2.1597211134883234) q[11];
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
ry(3.0793753853547057) q[0];
rz(0.9138009233400624) q[0];
ry(-1.5803902941033112) q[1];
rz(1.0252225674275333) q[1];
ry(1.8029838592678795) q[2];
rz(-2.339025637974096) q[2];
ry(0.8938646916041119) q[3];
rz(1.099047959851733) q[3];
ry(2.8829710620114466) q[4];
rz(-0.1028273129147962) q[4];
ry(-3.0623079918219607) q[5];
rz(-2.9236717313911704) q[5];
ry(3.137267676757152) q[6];
rz(1.3424497341067019) q[6];
ry(-3.132897778193624) q[7];
rz(2.6568200196614513) q[7];
ry(2.186456275403475) q[8];
rz(-1.9477482165040763) q[8];
ry(-3.135753859279893) q[9];
rz(-0.5509139322059068) q[9];
ry(-1.5640634121903445) q[10];
rz(-1.5000659324539418) q[10];
ry(-1.4740208931208854) q[11];
rz(-0.07116316288304514) q[11];
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
ry(2.1578826767258494) q[0];
rz(2.6571157931081952) q[0];
ry(1.6371603967417845) q[1];
rz(-2.8127391938026) q[1];
ry(-2.6661721988304103) q[2];
rz(-2.0900410058918313) q[2];
ry(-1.905643611957563) q[3];
rz(-1.536099464551482) q[3];
ry(1.480005757629426) q[4];
rz(-0.02263889063882201) q[4];
ry(-1.4491136936956561) q[5];
rz(-0.028927396915214842) q[5];
ry(3.13822891233356) q[6];
rz(-1.4673333853870647) q[6];
ry(-0.004294014210804775) q[7];
rz(0.2183395810738302) q[7];
ry(-0.006980835981543499) q[8];
rz(-2.7587484038205274) q[8];
ry(-0.01114185541428192) q[9];
rz(-0.054345129100111494) q[9];
ry(-1.384648109100195) q[10];
rz(-3.0711944254106753) q[10];
ry(-1.5850119965656135) q[11];
rz(3.048156305508185) q[11];
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
ry(-1.2519539263475779) q[0];
rz(1.1670512615745565) q[0];
ry(2.2257786616663364) q[1];
rz(-1.4110946592506481) q[1];
ry(1.4862251114511873) q[2];
rz(-1.3514485143585606) q[2];
ry(2.505261891855839) q[3];
rz(1.9449392681601811) q[3];
ry(2.2117070437968227) q[4];
rz(1.574225122250887) q[4];
ry(0.5950703496473074) q[5];
rz(-1.510612311476104) q[5];
ry(-0.011233531452341683) q[6];
rz(2.1426798388441766) q[6];
ry(1.5903636251864173) q[7];
rz(1.5743715597174281) q[7];
ry(1.4722878462884896) q[8];
rz(2.9185897361811293) q[8];
ry(1.5691880256805566) q[9];
rz(-1.5297951304592579) q[9];
ry(-1.5181014201267482) q[10];
rz(-1.989386928414187) q[10];
ry(3.094200899307558) q[11];
rz(-1.6987885563179308) q[11];
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
ry(-1.685681449690758) q[0];
rz(-1.2478389471402842) q[0];
ry(-2.8633978182870683) q[1];
rz(-2.2483175268750673) q[1];
ry(0.2454697985184269) q[2];
rz(-0.4608922972433987) q[2];
ry(-0.9148603759791847) q[3];
rz(-0.7255643764485677) q[3];
ry(1.573744247362649) q[4];
rz(-0.0032348120602636926) q[4];
ry(1.5654136229976614) q[5];
rz(3.1403369788307267) q[5];
ry(-2.1655421164766175) q[6];
rz(1.5747977603168202) q[6];
ry(2.5248743995395877) q[7];
rz(1.5718399715830755) q[7];
ry(1.5692289479996417) q[8];
rz(1.709630819825656) q[8];
ry(1.576325465624777) q[9];
rz(-2.6992476183003102) q[9];
ry(-0.491979235153082) q[10];
rz(1.897620050024649) q[10];
ry(-1.0556483933877585) q[11];
rz(1.764430078914214) q[11];
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
ry(1.3871886670932625) q[0];
rz(0.48193588564705436) q[0];
ry(1.271462660403162) q[1];
rz(2.1514086508188957) q[1];
ry(-0.18324449315774177) q[2];
rz(-0.6429532912846252) q[2];
ry(-1.6403694890062122) q[3];
rz(2.2120175699911586) q[3];
ry(1.5727905250602494) q[4];
rz(1.5777421702122048) q[4];
ry(-1.574251458024206) q[5];
rz(-2.931346898134467) q[5];
ry(-1.5639774428831936) q[6];
rz(3.1387263472396785) q[6];
ry(-1.576616159013959) q[7];
rz(-0.0020241297529670636) q[7];
ry(-0.05098974285303801) q[8];
rz(2.978640489532278) q[8];
ry(-0.000978112207222992) q[9];
rz(1.1736961653333617) q[9];
ry(2.6822083567608384) q[10];
rz(-2.1944083980753817) q[10];
ry(-1.6478175016083194) q[11];
rz(-1.4891818976332365) q[11];
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
ry(-2.913153260654376) q[0];
rz(-2.4790596164149585) q[0];
ry(-1.4222631781127486) q[1];
rz(2.9771201523425934) q[1];
ry(2.4160362190845714) q[2];
rz(2.270825707394697) q[2];
ry(2.8544909771990934) q[3];
rz(0.7702898414540531) q[3];
ry(-2.9540893421397048) q[4];
rz(-1.0907237919754875) q[4];
ry(-3.1311217233991506) q[5];
rz(-0.3138019444541156) q[5];
ry(1.6365462058171614) q[6];
rz(-1.6173877343537981) q[6];
ry(1.5633848766679028) q[7];
rz(2.0943494624122003) q[7];
ry(-3.0367965077913097) q[8];
rz(-1.5980237758138869) q[8];
ry(3.129042995672172) q[9];
rz(-3.096604500336253) q[9];
ry(-0.06009304209072475) q[10];
rz(-0.35174871003171937) q[10];
ry(1.5637648395045285) q[11];
rz(1.1516378270591792) q[11];
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
ry(0.8707382119544503) q[0];
rz(1.7235569535007633) q[0];
ry(1.0360732586891979) q[1];
rz(-2.860016833329525) q[1];
ry(0.6539476721341719) q[2];
rz(1.3715929590379434) q[2];
ry(0.9190668965748771) q[3];
rz(1.2066568020036001) q[3];
ry(0.013948472761012631) q[4];
rz(-0.24988282792409855) q[4];
ry(-3.131980823138973) q[5];
rz(1.0432163382519475) q[5];
ry(0.0013870992547731283) q[6];
rz(1.9710864076961858) q[6];
ry(-3.136671623460489) q[7];
rz(-1.0312279854636701) q[7];
ry(-1.4158897883031147) q[8];
rz(3.0660813817551666) q[8];
ry(-1.5680179064360973) q[9];
rz(1.5948390070431457) q[9];
ry(2.656741214162874) q[10];
rz(-2.0978616929925327) q[10];
ry(2.617833590553416) q[11];
rz(-1.9083736806093974) q[11];
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
ry(-2.6034235373587604) q[0];
rz(-1.347929135793465) q[0];
ry(-2.263846792230258) q[1];
rz(-2.4191156585886153) q[1];
ry(-1.4863757096381) q[2];
rz(-0.3791667265798227) q[2];
ry(0.8756004165340663) q[3];
rz(-1.5132783459863781) q[3];
ry(0.008836301985778226) q[4];
rz(2.863270005663315) q[4];
ry(0.5426236678443415) q[5];
rz(-2.8429761980720083) q[5];
ry(3.138094584480302) q[6];
rz(1.9299677974801543) q[6];
ry(2.829442145565113) q[7];
rz(-1.560671282313793) q[7];
ry(1.435935103247239) q[8];
rz(-1.564651535403677) q[8];
ry(-1.8122985453021894) q[9];
rz(-1.211287268199073) q[9];
ry(1.609688311719779) q[10];
rz(3.1334509736952487) q[10];
ry(-3.112298080761818) q[11];
rz(-0.06632551255536079) q[11];
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
ry(-2.921338676548538) q[0];
rz(-2.515997960317) q[0];
ry(-1.8215366472239196) q[1];
rz(3.049801812606632) q[1];
ry(-1.4532296380339067) q[2];
rz(-0.009562810197343712) q[2];
ry(1.4544218768444659) q[3];
rz(1.8437029985913007) q[3];
ry(-3.138773048893159) q[4];
rz(-1.622019464125665) q[4];
ry(-3.128014164910344) q[5];
rz(2.2762404249029435) q[5];
ry(-1.5738187545994893) q[6];
rz(-2.5427908576049547) q[6];
ry(1.5746402425651658) q[7];
rz(3.1414632136014835) q[7];
ry(1.541744140196258) q[8];
rz(-0.00037784889180248724) q[8];
ry(-0.0007872887056193396) q[9];
rz(2.402670084874383) q[9];
ry(1.9042847925594195) q[10];
rz(-1.5390102320323995) q[10];
ry(1.5719322619314287) q[11];
rz(-1.5515866017296338) q[11];
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
ry(-0.9836545752840524) q[0];
rz(-2.131873504004256) q[0];
ry(1.0415634900402841) q[1];
rz(-2.4705223590339065) q[1];
ry(1.9096775035336317) q[2];
rz(-0.3619417834125107) q[2];
ry(-0.9874062456316899) q[3];
rz(-0.7633283343955304) q[3];
ry(-0.0014357018147903658) q[4];
rz(2.883490178467593) q[4];
ry(0.005628324211364167) q[5];
rz(1.1329887563452203) q[5];
ry(-0.022272025135387175) q[6];
rz(-0.07098773160202132) q[6];
ry(-1.835253541609912) q[7];
rz(3.137571490198577) q[7];
ry(1.571592219409241) q[8];
rz(-0.34339204224077624) q[8];
ry(-3.1385071494627126) q[9];
rz(-0.2829994440921926) q[9];
ry(1.5738461618381256) q[10];
rz(-1.6428920628110282) q[10];
ry(2.3598895113238623) q[11];
rz(0.005839304438091455) q[11];
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
ry(-2.132844328758133) q[0];
rz(-2.049244340702652) q[0];
ry(1.892645891801399) q[1];
rz(-2.3129912526835428) q[1];
ry(-2.448389767894032) q[2];
rz(-2.2582192005176633) q[2];
ry(2.4711497197870624) q[3];
rz(-0.7713819334384185) q[3];
ry(-3.1323524548203348) q[4];
rz(-1.891984985717503) q[4];
ry(3.023346977292169) q[5];
rz(1.5606087033121083) q[5];
ry(3.138264239101906) q[6];
rz(-2.6140792720360366) q[6];
ry(-1.5692920283950829) q[7];
rz(-3.139221175569277) q[7];
ry(7.586501294251755e-05) q[8];
rz(-1.3668858235675803) q[8];
ry(0.00028270172731145705) q[9];
rz(-0.25727398457538886) q[9];
ry(0.00830771487247528) q[10];
rz(-1.5218926788227363) q[10];
ry(1.6198403350278046) q[11];
rz(-1.5233640379334803) q[11];
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
ry(-0.22706363796419812) q[0];
rz(2.8478883266416823) q[0];
ry(2.3315708842538507) q[1];
rz(-2.3587756363124486) q[1];
ry(1.3416077397380866) q[2];
rz(-2.495927399438455) q[2];
ry(-2.1233789566921413) q[3];
rz(0.6641467523475124) q[3];
ry(-3.1392806872694043) q[4];
rz(3.0847767557478885) q[4];
ry(0.38396805533391815) q[5];
rz(3.1265891977664837) q[5];
ry(1.5746508388304625) q[6];
rz(-0.6630308034153033) q[6];
ry(-1.820889730560697) q[7];
rz(-3.0547348452413914) q[7];
ry(0.034623825625112244) q[8];
rz(-3.0119336765876183) q[8];
ry(-0.041417527250173336) q[9];
rz(0.15270119735979026) q[9];
ry(1.5529334122623402) q[10];
rz(2.4164420705823777) q[10];
ry(1.5976758887031144) q[11];
rz(1.0366107048055646) q[11];
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
ry(-2.4086179058879797) q[0];
rz(0.2940756819943777) q[0];
ry(1.5955360582188458) q[1];
rz(-0.9588785505376434) q[1];
ry(3.139678098595459) q[2];
rz(-1.2350632859052217) q[2];
ry(1.8651675963738423) q[3];
rz(-1.5315624540588715) q[3];
ry(1.5709830376225635) q[4];
rz(-3.1413821291235995) q[4];
ry(-1.5740522747550871) q[5];
rz(-0.004926819914054354) q[5];
ry(-3.135964306788129) q[6];
rz(-2.233747560681188) q[6];
ry(-0.006869255895354494) q[7];
rz(1.4850062614889898) q[7];
ry(3.141465885168864) q[8];
rz(0.02615411258246869) q[8];
ry(3.101244406459921) q[9];
rz(-1.5976439281686998) q[9];
ry(0.07537381902415217) q[10];
rz(0.7300461816410575) q[10];
ry(0.007168827811057699) q[11];
rz(1.5397799948938673) q[11];
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
ry(1.627864089785267) q[0];
rz(-0.6416443065072706) q[0];
ry(1.5548178467880533) q[1];
rz(1.5282846445750584) q[1];
ry(-1.2465832505211978) q[2];
rz(1.5859057148506066) q[2];
ry(0.13378052855379963) q[3];
rz(-0.09683855837241587) q[3];
ry(1.5726666174514925) q[4];
rz(-5.409734116019311e-05) q[4];
ry(1.57161066370118) q[5];
rz(-1.566976672496517) q[5];
ry(1.5722221511393668) q[6];
rz(0.000941902130636917) q[6];
ry(1.5708840544601161) q[7];
rz(1.593873681462199) q[7];
ry(0.03443085200791367) q[8];
rz(3.1043251920595427) q[8];
ry(-1.5685561078401058) q[9];
rz(-2.626033408717023) q[9];
ry(1.5497065395827763) q[10];
rz(-1.5465994266038086) q[10];
ry(-1.5744214380043768) q[11];
rz(0.04960524086270369) q[11];
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
ry(0.052389224954915825) q[0];
rz(-1.393112921525974) q[0];
ry(-3.0936370108006526) q[1];
rz(-2.78087878487112) q[1];
ry(1.4662045865777609) q[2];
rz(-1.5506351845431623) q[2];
ry(0.010545798111143867) q[3];
rz(2.0647385969782235) q[3];
ry(-1.5812683993768886) q[4];
rz(-0.45976408440439975) q[4];
ry(-1.5663879228969755) q[5];
rz(-1.163998793056479) q[5];
ry(-1.5713452898862101) q[6];
rz(2.6838293145834498) q[6];
ry(-3.141028668033119) q[7];
rz(2.004019267280488) q[7];
ry(-1.5686131046773053) q[8];
rz(1.11353196267619) q[8];
ry(-3.1407756923928494) q[9];
rz(2.457331976483632) q[9];
ry(1.5708151334424398) q[10];
rz(2.6844953864915486) q[10];
ry(1.5710575003274458) q[11];
rz(1.9844875678384852) q[11];