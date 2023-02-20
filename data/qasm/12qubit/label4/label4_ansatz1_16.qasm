OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
ry(0.5477237123848878) q[0];
rz(1.0103935866253693) q[0];
ry(-2.2228264678612084) q[1];
rz(0.5181613048273448) q[1];
ry(1.0153273274735577) q[2];
rz(3.049052944810741) q[2];
ry(-0.037323030044046135) q[3];
rz(0.7193107634623336) q[3];
ry(-1.6240094818291846) q[4];
rz(3.110528352794485) q[4];
ry(-0.9193168272319299) q[5];
rz(0.7620426243964495) q[5];
ry(1.9268749481033476) q[6];
rz(-3.138697253174195) q[6];
ry(-0.0024221778083219903) q[7];
rz(3.134948819101535) q[7];
ry(2.2232303873175194) q[8];
rz(0.6589455830191002) q[8];
ry(2.3359833190561643) q[9];
rz(0.5654060037825204) q[9];
ry(2.8369980106254036) q[10];
rz(3.0519464910767495) q[10];
ry(-2.5188776651646845) q[11];
rz(-2.7795698385806245) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-2.6777277894741447) q[0];
rz(1.4752282858332968) q[0];
ry(1.4591948812528774) q[1];
rz(-1.011504159977667) q[1];
ry(-2.030149383939099) q[2];
rz(3.122226208029178) q[2];
ry(0.2648807943158973) q[3];
rz(1.7672030071267582) q[3];
ry(2.113440175207114) q[4];
rz(-0.0742191125057431) q[4];
ry(0.0404088051902649) q[5];
rz(-0.09914638439921665) q[5];
ry(2.4441530180313045) q[6];
rz(0.6246494256990872) q[6];
ry(-2.986083898300278) q[7];
rz(0.18481510819002367) q[7];
ry(-1.2545241388350465) q[8];
rz(0.45246543909498455) q[8];
ry(3.065814843437199) q[9];
rz(-0.32923207985215525) q[9];
ry(-3.1318863414076943) q[10];
rz(-0.07713745247241797) q[10];
ry(-0.7874962806232111) q[11];
rz(-2.346845174588267) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-3.101565704207184) q[0];
rz(1.499895829763206) q[0];
ry(-2.951542942953329) q[1];
rz(-0.6237478244203807) q[1];
ry(-2.3733856592611287) q[2];
rz(2.096804596594766) q[2];
ry(0.0010634861944440743) q[3];
rz(-1.816830292704477) q[3];
ry(-0.06452875626568133) q[4];
rz(1.3847749114367973) q[4];
ry(-2.083997914375236) q[5];
rz(-0.290524544244154) q[5];
ry(0.009047592349173758) q[6];
rz(0.2703900982192976) q[6];
ry(-0.0008222613832173587) q[7];
rz(2.073930537568277) q[7];
ry(1.1347713016401126) q[8];
rz(0.524004366165841) q[8];
ry(-1.5178028731254276) q[9];
rz(2.26935958217536) q[9];
ry(1.811582125944832) q[10];
rz(1.7340682380961718) q[10];
ry(2.914339867889257) q[11];
rz(2.8821287493779337) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.7536493715102095) q[0];
rz(-0.7837129362804003) q[0];
ry(2.086370180078595) q[1];
rz(-2.6674625151838245) q[1];
ry(-2.6032888565512633) q[2];
rz(-1.2601907349809336) q[2];
ry(-2.655739503765331) q[3];
rz(-1.934471009734706) q[3];
ry(2.889251431791641) q[4];
rz(1.2430530594718707) q[4];
ry(0.23540615577326385) q[5];
rz(2.59981383359262) q[5];
ry(-1.288022813750326) q[6];
rz(-0.525201445915866) q[6];
ry(0.08300352154139645) q[7];
rz(-2.1154138732307377) q[7];
ry(-1.5316743503218166) q[8];
rz(-0.518117251536451) q[8];
ry(0.19747181973938593) q[9];
rz(2.7503947856483184) q[9];
ry(1.2232821166775363) q[10];
rz(-1.4016894788990946) q[10];
ry(2.3087488841232413) q[11];
rz(-3.03163701322823) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.9430449184835233) q[0];
rz(1.12249105261066) q[0];
ry(0.38438080316924955) q[1];
rz(2.325462756025526) q[1];
ry(2.022200021563023) q[2];
rz(-0.5419064771656128) q[2];
ry(-0.00393257255901454) q[3];
rz(-0.7651932484905863) q[3];
ry(-0.07658125080792995) q[4];
rz(1.914686459948224) q[4];
ry(-1.8159417632781085) q[5];
rz(1.656692007767126) q[5];
ry(-1.6592935820059154) q[6];
rz(-0.4080514379973896) q[6];
ry(-3.1402345159862337) q[7];
rz(-1.8764767321822768) q[7];
ry(0.07559867977507119) q[8];
rz(-0.8794132639178339) q[8];
ry(0.09358566056075279) q[9];
rz(-0.900917195750468) q[9];
ry(-0.015658935749586836) q[10];
rz(-1.9168241687512226) q[10];
ry(3.0690251617473985) q[11];
rz(-0.5254566095195575) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.7069175432913615) q[0];
rz(2.3001954373178033) q[0];
ry(0.04501923883870734) q[1];
rz(1.0590779570923565) q[1];
ry(-0.9245308724447218) q[2];
rz(0.43441525101064177) q[2];
ry(1.3569004583895328) q[3];
rz(-1.4475993807304197) q[3];
ry(0.9696101733200394) q[4];
rz(0.008176439042244032) q[4];
ry(0.8689801267125823) q[5];
rz(1.6449495770246525) q[5];
ry(-0.08269043105505368) q[6];
rz(0.026516175582912634) q[6];
ry(1.5561833936426273) q[7];
rz(-1.6976465546049402) q[7];
ry(-1.2281852297173659) q[8];
rz(-0.01047745532120987) q[8];
ry(1.408207303545285) q[9];
rz(2.9648438156039996) q[9];
ry(-2.3215883072783225) q[10];
rz(1.6470142220148345) q[10];
ry(-2.1458924247551603) q[11];
rz(-1.2084498792056622) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.5638526515585667) q[0];
rz(-1.8534793872861268) q[0];
ry(0.4481755434180288) q[1];
rz(-1.166203134489871) q[1];
ry(1.8496687661141564) q[2];
rz(-2.427617284473195) q[2];
ry(-1.571847639937991) q[3];
rz(-1.5775379099530384) q[3];
ry(3.1394589398375365) q[4];
rz(0.03567351286134545) q[4];
ry(-0.0378867290378785) q[5];
rz(-1.6658490664899288) q[5];
ry(-3.139730610170474) q[6];
rz(-0.16824171416330638) q[6];
ry(0.03075627802901436) q[7];
rz(-3.0438794213179174) q[7];
ry(-0.2794717860346258) q[8];
rz(-1.1990722197607369) q[8];
ry(-1.1490367577767753) q[9];
rz(-2.010769102455742) q[9];
ry(0.04853620073276588) q[10];
rz(0.196759904884299) q[10];
ry(-2.4648590951752083) q[11];
rz(-2.5395625925042733) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.552289325472563) q[0];
rz(0.20149080622317062) q[0];
ry(2.9024302521061367) q[1];
rz(-2.477283816971202) q[1];
ry(-3.1400018472212765) q[2];
rz(0.7311957035195124) q[2];
ry(1.522589844982197) q[3];
rz(0.9174914224036951) q[3];
ry(-1.6074636958739232) q[4];
rz(0.00045266034172490355) q[4];
ry(-0.8825442408462463) q[5];
rz(-0.8826353021446658) q[5];
ry(-2.996516956794087) q[6];
rz(-2.893587969853382) q[6];
ry(2.4090897102280264) q[7];
rz(-3.1366348377468736) q[7];
ry(-0.3877228791344404) q[8];
rz(-2.8613253422259746) q[8];
ry(-0.9816759400946985) q[9];
rz(1.6284991139907294) q[9];
ry(0.04946601939623019) q[10];
rz(-1.7694709565993743) q[10];
ry(1.4053054509667473) q[11];
rz(-2.945925017632478) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.3572336111604377) q[0];
rz(1.3091903070100532) q[0];
ry(-1.6618843081802226) q[1];
rz(0.9592258103813297) q[1];
ry(1.566850640441609) q[2];
rz(-1.4171910848596525) q[2];
ry(0.0005466094419821488) q[3];
rz(-2.756172253447277) q[3];
ry(1.5601219795318422) q[4];
rz(1.6495087951869793) q[4];
ry(-1.5793211782778869) q[5];
rz(3.138953884468033) q[5];
ry(-1.4560855497346168) q[6];
rz(-0.0015770063138298201) q[6];
ry(1.6816735279599244) q[7];
rz(3.1357328193258267) q[7];
ry(-2.989881571989875) q[8];
rz(-2.553392247720438) q[8];
ry(1.0434199848032824) q[9];
rz(-2.8639649868104105) q[9];
ry(-3.1368795154169113) q[10];
rz(-1.2949441465756601) q[10];
ry(1.0269951734365304) q[11];
rz(0.4513242107022278) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-0.3840895443167676) q[0];
rz(-2.187450006080024) q[0];
ry(2.271660430476617) q[1];
rz(0.7545186902523399) q[1];
ry(1.6379159757666786) q[2];
rz(-2.366286227847748) q[2];
ry(-3.1411067858492454) q[3];
rz(-2.8573499618000215) q[3];
ry(2.976651311181288) q[4];
rz(1.694383850484705) q[4];
ry(-1.5018181114154396) q[5];
rz(2.981123220986506) q[5];
ry(-1.5541438010894626) q[6];
rz(0.16022719826534795) q[6];
ry(1.5225292821804495) q[7];
rz(3.140810236587353) q[7];
ry(2.0893712023438358) q[8];
rz(-3.1370624865976175) q[8];
ry(-2.62986907341256) q[9];
rz(0.2511759143140341) q[9];
ry(2.1650845340755627) q[10];
rz(0.6994710754321777) q[10];
ry(-2.110281979687124) q[11];
rz(-1.4186728140952198) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-3.0939276417642354) q[0];
rz(-2.821614276415181) q[0];
ry(-3.109443294742743) q[1];
rz(-1.627850705811665) q[1];
ry(-0.013587731993335517) q[2];
rz(-3.114715733917663) q[2];
ry(-0.02358909345058091) q[3];
rz(2.4150292511657763) q[3];
ry(-1.5741625852267136) q[4];
rz(1.7566480239027182) q[4];
ry(-3.0540543348405977) q[5];
rz(0.9546131243551733) q[5];
ry(-0.6838963987692404) q[6];
rz(-1.3547172992120733) q[6];
ry(-3.019083161697058) q[7];
rz(0.3033297015651594) q[7];
ry(2.298303130842718) q[8];
rz(-3.1262713550004486) q[8];
ry(-2.2488605024891077) q[9];
rz(-2.8378588977900763) q[9];
ry(-3.118499987723451) q[10];
rz(-2.332826982440593) q[10];
ry(-2.194366284756964) q[11];
rz(1.6874467255563628) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(2.3507641655025955) q[0];
rz(0.319409004369871) q[0];
ry(2.241386477713602) q[1];
rz(-0.9995934161460163) q[1];
ry(-2.9289489594738884) q[2];
rz(-0.7255336267927577) q[2];
ry(1.5630505833164898) q[3];
rz(0.24180697478578442) q[3];
ry(-3.1402329925164754) q[4];
rz(1.7567204539308152) q[4];
ry(-0.02328806399571981) q[5];
rz(2.0264050485146567) q[5];
ry(-0.007416533150093253) q[6];
rz(-1.9842899869926247) q[6];
ry(-0.01377748224048625) q[7];
rz(-0.304054331234071) q[7];
ry(-2.294533033290654) q[8];
rz(3.0287866768892653) q[8];
ry(3.1209192348726957) q[9];
rz(1.8751218014609379) q[9];
ry(0.05381594700957391) q[10];
rz(2.5718773737233356) q[10];
ry(-2.6936815726897905) q[11];
rz(-2.1680539333313833) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(3.106725212181919) q[0];
rz(3.1335994221799552) q[0];
ry(1.596611185158923) q[1];
rz(-1.3272587241923464) q[1];
ry(1.5857324181967927) q[2];
rz(0.4565529273241073) q[2];
ry(2.905115927842729) q[3];
rz(0.2990928869369275) q[3];
ry(1.5775468933121406) q[4];
rz(-3.0934583319831765) q[4];
ry(-1.6148487562174731) q[5];
rz(-0.013712926308613047) q[5];
ry(-2.2380367449016836) q[6];
rz(2.9939833544910273) q[6];
ry(1.5872247883667572) q[7];
rz(-0.002246922183341516) q[7];
ry(2.7617459680980083) q[8];
rz(1.959632092801273) q[8];
ry(1.5881929562711037) q[9];
rz(3.097816359742787) q[9];
ry(0.008367872488181936) q[10];
rz(-1.0594214187454116) q[10];
ry(-0.8973476029099443) q[11];
rz(1.9160228220593327) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.5524320421991513) q[0];
rz(0.005246854834167536) q[0];
ry(1.5678666104729466) q[1];
rz(0.564330600143542) q[1];
ry(0.00020325836308018202) q[2];
rz(-0.5848819712885743) q[2];
ry(-0.8498550814561652) q[3];
rz(-3.086540360982079) q[3];
ry(-1.2452745076695813) q[4];
rz(2.1285817218152987) q[4];
ry(-2.064225215229734) q[5];
rz(3.077376725552017) q[5];
ry(3.0755917776809545) q[6];
rz(3.122696332318962) q[6];
ry(-1.5821267169662296) q[7];
rz(1.7665126198873131) q[7];
ry(3.1411747175713405) q[8];
rz(0.48654553700978087) q[8];
ry(1.5951630026596755) q[9];
rz(1.5518696790977407) q[9];
ry(1.5939697458044773) q[10];
rz(-3.1403119545115104) q[10];
ry(1.26566547385878) q[11];
rz(0.4061523688149701) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.4800627842459984) q[0];
rz(-2.562393071623576) q[0];
ry(3.1409285301689054) q[1];
rz(0.9521799280624528) q[1];
ry(1.5803790668841216) q[2];
rz(-0.0017825460871314647) q[2];
ry(-1.676131948326753) q[3];
rz(-2.679064403714524) q[3];
ry(1.0875782179636388) q[4];
rz(-1.0040751664964835) q[4];
ry(0.699541538810033) q[5];
rz(0.13846919244645228) q[5];
ry(-1.8081454691024774) q[6];
rz(3.1003776467508057) q[6];
ry(0.0031681745825201824) q[7];
rz(-1.7669759404132235) q[7];
ry(-0.3748053446023777) q[8];
rz(-0.003360110938741751) q[8];
ry(1.5698265496462955) q[9];
rz(-1.5068042208609391) q[9];
ry(-1.529449637051374) q[10];
rz(-0.006305086636445801) q[10];
ry(3.1058686967157514) q[11];
rz(2.60606386388179) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-3.1401714357452724) q[0];
rz(1.4785873568067105) q[0];
ry(-0.49875253211626597) q[1];
rz(2.711274333969533) q[1];
ry(1.5733485881893516) q[2];
rz(1.5868168096869868) q[2];
ry(0.0010921154790839438) q[3];
rz(-0.44232044384393004) q[3];
ry(0.0018792742180639976) q[4];
rz(-0.8566941947150185) q[4];
ry(3.14119017609663) q[5];
rz(-3.0914715831314123) q[5];
ry(3.0870400612809714) q[6];
rz(3.0990360820923826) q[6];
ry(-1.579328681913583) q[7];
rz(2.904214599355255) q[7];
ry(2.684712844994511) q[8];
rz(1.9673468013183808) q[8];
ry(-3.1370915622173814) q[9];
rz(-3.0362385083999084) q[9];
ry(-2.3020514328073447) q[10];
rz(3.1288967676915607) q[10];
ry(-1.468833142041313) q[11];
rz(3.136776391865663) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(-1.6823031109892983) q[0];
rz(3.0051291917101066) q[0];
ry(1.5706207048492296) q[1];
rz(-3.1402231374185305) q[1];
ry(3.1410489652747553) q[2];
rz(2.927443389973591) q[2];
ry(-1.5577947924220732) q[3];
rz(-2.4408638828819305) q[3];
ry(1.0578447950718726) q[4];
rz(1.9904786173089963) q[4];
ry(0.827856747135873) q[5];
rz(0.18260143788514926) q[5];
ry(1.5575202467789477) q[6];
rz(-0.08040852765476411) q[6];
ry(-0.48095051079959283) q[7];
rz(-0.8680583891861996) q[7];
ry(1.9608855133087864) q[8];
rz(1.6869634862479754) q[8];
ry(0.0022994698511862306) q[9];
rz(-2.1208314033865134) q[9];
ry(2.1400259464877744) q[10];
rz(3.136174113518683) q[10];
ry(1.5388444615625332) q[11];
rz(1.944151667870849) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.0050163393119499805) q[0];
rz(1.5107499833971716) q[0];
ry(-1.9332350244374004) q[1];
rz(0.046638288372021885) q[1];
ry(-1.5333288357709831) q[2];
rz(0.908810678364964) q[2];
ry(0.6567470937573818) q[3];
rz(0.9680851521659628) q[3];
ry(1.4271029942293703) q[4];
rz(-2.168714782566955) q[4];
ry(0.37604633059420617) q[5];
rz(3.0248340998458274) q[5];
ry(3.139450641271506) q[6];
rz(-2.9733197595005016) q[6];
ry(0.00016021321795950836) q[7];
rz(1.124944303433038) q[7];
ry(-3.140353284031848) q[8];
rz(-1.457980745920516) q[8];
ry(0.0004333991582755317) q[9];
rz(0.5115451457370161) q[9];
ry(-1.6015826881526973) q[10];
rz(3.1186440056494877) q[10];
ry(3.1045123501527265) q[11];
rz(-1.228399935317304) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(1.3409200415935578) q[0];
rz(-1.2189131624844638) q[0];
ry(0.0050486152599376055) q[1];
rz(-1.6187309573429667) q[1];
ry(-0.0012157344053436866) q[2];
rz(0.5078549247785107) q[2];
ry(-3.1384977739461823) q[3];
rz(-0.013480996515221562) q[3];
ry(-0.0002341878021110648) q[4];
rz(-2.5232111945971702) q[4];
ry(-3.1390811616729324) q[5];
rz(-1.569590443347268) q[5];
ry(-3.1287137018983664) q[6];
rz(1.8200493936863313) q[6];
ry(1.9623353758935682) q[7];
rz(-1.4530941013048215) q[7];
ry(1.961302785844497) q[8];
rz(0.3981540808873696) q[8];
ry(-1.5713860102154482) q[9];
rz(1.5714946762162574) q[9];
ry(-2.3858892448487343) q[10];
rz(0.005630260262309505) q[10];
ry(-2.381333324073812) q[11];
rz(2.3851868473077427) q[11];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[8],q[9];
cz q[10],q[11];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
cz q[7],q[8];
cz q[9],q[10];
ry(0.009597560386232615) q[0];
rz(1.8738992138435266) q[0];
ry(-1.5697476266661774) q[1];
rz(2.8335101026929115) q[1];
ry(-2.9091952525377622) q[2];
rz(-1.5885099308271273) q[2];
ry(1.1670326023193693) q[3];
rz(-2.3092241438583176) q[3];
ry(1.7064059127924054) q[4];
rz(-2.8444682509846797) q[4];
ry(-1.6140657974797463) q[5];
rz(-1.639294499056347) q[5];
ry(-1.570331733943743) q[6];
rz(1.6768933436280378) q[6];
ry(1.5702264636163221) q[7];
rz(-3.110030255446087) q[7];
ry(-1.5722513915977068) q[8];
rz(0.11634039004046492) q[8];
ry(1.5705210634922813) q[9];
rz(0.261175780751642) q[9];
ry(-1.5687801419479677) q[10];
rz(-3.042479895048211) q[10];
ry(0.03351839255948737) q[11];
rz(-0.8497421933752698) q[11];