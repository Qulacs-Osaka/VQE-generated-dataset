OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.10905041170756694) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.0681359598565292) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.06309878927313624) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.04633867373743236) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.11186299550211765) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.06173096680037375) q[3];
cx q[2],q[3];
rx(-0.02371581066556729) q[0];
rz(-0.02032236541540961) q[0];
rx(-0.1196079155751163) q[1];
rz(-0.10796468756186188) q[1];
rx(-0.05304541888637512) q[2];
rz(-0.08648060633385846) q[2];
rx(-0.11326368283919977) q[3];
rz(-0.025511717326804118) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.13644075531642583) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.019534137337645953) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.09030283127143689) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.01889239239865431) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.07535282280478425) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.11915698577832186) q[3];
cx q[2],q[3];
rx(-0.06236744491547937) q[0];
rz(-0.097181450053588) q[0];
rx(-0.0831217438459862) q[1];
rz(-0.11314482326010664) q[1];
rx(0.03231077922597549) q[2];
rz(-0.07893920571765402) q[2];
rx(-0.07741037027716637) q[3];
rz(-0.038218775729011056) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.19360943511340675) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.08335797738271827) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.05770773625310126) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.041975130258652196) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.06424514567041892) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.08690334054889126) q[3];
cx q[2],q[3];
rx(-0.12852008121586514) q[0];
rz(-0.07636652964697697) q[0];
rx(-0.0771568287532525) q[1];
rz(-0.0777431572009616) q[1];
rx(-0.018610996578311504) q[2];
rz(-0.08488128927790506) q[2];
rx(-0.16548344537291904) q[3];
rz(-0.007252134193755817) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.18871722590232115) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.05068986148972792) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.028038617757210376) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.07434703942852501) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.02699144852998395) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.0538810589043061) q[3];
cx q[2],q[3];
rx(-0.10531159592796951) q[0];
rz(-0.0012103788105216951) q[0];
rx(-0.04091573221821652) q[1];
rz(-0.07132998808688205) q[1];
rx(-0.027978070501267955) q[2];
rz(-0.1159338408840301) q[2];
rx(-0.1193858874526307) q[3];
rz(-0.016020010898489694) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.14110080948208956) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.04834970236678437) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.06360861096428203) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.11073905281969326) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.012666914156825878) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.011804340369420597) q[3];
cx q[2],q[3];
rx(-0.13079406904296564) q[0];
rz(0.015003515469033907) q[0];
rx(-0.05258611735966189) q[1];
rz(-0.05138649424451361) q[1];
rx(-0.03902812591918134) q[2];
rz(-0.061276431580200554) q[2];
rx(-0.11725447279788828) q[3];
rz(-0.021113759269184452) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1375818276804088) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.04363607734162297) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.03705021919009809) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.06163400057972443) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.007681069878557182) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.04583561319768154) q[3];
cx q[2],q[3];
rx(-0.14340197852522435) q[0];
rz(-0.05967473182843555) q[0];
rx(0.011418105980362513) q[1];
rz(-0.062203702764608806) q[1];
rx(0.02306777159858606) q[2];
rz(-0.03955108277840159) q[2];
rx(-0.12602285570277022) q[3];
rz(-0.07070184522533013) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.12173433552958954) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.00984816225316672) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.026433981986611062) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1112438371053587) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.008147013808925282) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.04062353958703254) q[3];
cx q[2],q[3];
rx(-0.1858637726097858) q[0];
rz(-0.0011549066825267554) q[0];
rx(-0.0649190306647266) q[1];
rz(-0.035946222706146516) q[1];
rx(-0.022100040729271374) q[2];
rz(-0.09240987502315333) q[2];
rx(-0.16396580081213463) q[3];
rz(0.0020167650139936207) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.16390223079325614) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.014074235569805317) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.024374655719468237) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.10545599112272591) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.030132906658617745) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.03757114349045716) q[3];
cx q[2],q[3];
rx(-0.1564674862874394) q[0];
rz(-0.0035448767754413697) q[0];
rx(-0.015096312662337507) q[1];
rz(-0.09145806167067048) q[1];
rx(0.02607088169322572) q[2];
rz(-0.014923078699768062) q[2];
rx(-0.15325109115795693) q[3];
rz(-0.0633261806891337) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.10855795336168006) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.028605337549867958) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.019417500550658284) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.12324974043026733) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.0033268525067309622) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.028502218376056696) q[3];
cx q[2],q[3];
rx(-0.1911865842985744) q[0];
rz(-0.06189435134329164) q[0];
rx(-0.04162934736110542) q[1];
rz(-0.12899361341752452) q[1];
rx(0.02607478435542333) q[2];
rz(-0.06633734392204468) q[2];
rx(-0.13070550692043892) q[3];
rz(0.004078741764195368) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.10299552296824084) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.06354510203990782) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.0024161175732267262) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.14319817318094824) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.025451330083128932) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.06813077851032343) q[3];
cx q[2],q[3];
rx(-0.14145141103331393) q[0];
rz(-0.07033812055836625) q[0];
rx(-0.047612852117836564) q[1];
rz(-0.08422258549450005) q[1];
rx(0.0077139197798861735) q[2];
rz(-0.0030305296015438548) q[2];
rx(-0.12010544981694428) q[3];
rz(-0.021257900712077653) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.08456532797204261) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.04045370784825141) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.009897764953233242) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.09593233679604994) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.007124325268616046) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.1045344939990571) q[3];
cx q[2],q[3];
rx(-0.16515803287263822) q[0];
rz(0.019761387789431498) q[0];
rx(-0.005840871896385747) q[1];
rz(-0.0624600794256687) q[1];
rx(0.038654148446352045) q[2];
rz(-0.021572195633189333) q[2];
rx(-0.15139287274528113) q[3];
rz(-0.00660183695314959) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.033152938661254225) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.0469485205672462) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.04914776813364057) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.051934763818757765) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.045364906173862345) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.042423310445681296) q[3];
cx q[2],q[3];
rx(-0.15191600974930772) q[0];
rz(0.001445539732998302) q[0];
rx(0.019395206259110186) q[1];
rz(-0.12415669389904001) q[1];
rx(-0.00024283523490965353) q[2];
rz(-0.08781738760172292) q[2];
rx(-0.1490893686769023) q[3];
rz(-0.022979681420672825) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0895820306577199) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.011762483524245741) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.02017885535449454) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.07805708369238719) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(0.007213465640097053) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.04467763815682766) q[3];
cx q[2],q[3];
rx(-0.19060357320326538) q[0];
rz(-0.00015437958625222966) q[0];
rx(-0.015670356081578168) q[1];
rz(-0.12715283040502154) q[1];
rx(-0.015016348486923622) q[2];
rz(-0.053523568639210664) q[2];
rx(-0.21957753751008732) q[3];
rz(-0.04193211414890994) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.06638065783366082) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.013269340894918278) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.021072155912517924) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.05514882273290927) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.04934529492012993) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.0011187116962068425) q[3];
cx q[2],q[3];
rx(-0.19703748008801997) q[0];
rz(-0.008499705373286884) q[0];
rx(0.014677993565014386) q[1];
rz(-0.11164574636806746) q[1];
rx(-0.028166663354672994) q[2];
rz(-0.032483184930152766) q[2];
rx(-0.2100502342670576) q[3];
rz(0.04114414760072364) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.11885577349608557) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.008921354599463137) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.041818771220320766) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.06043259141387374) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.020670659746919423) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.01629584034510972) q[3];
cx q[2],q[3];
rx(-0.11380672820983559) q[0];
rz(0.07450658591853296) q[0];
rx(0.02392319348569753) q[1];
rz(-0.07430989397504417) q[1];
rx(0.0020149795023334783) q[2];
rz(-0.13693946891067663) q[2];
rx(-0.131116595668183) q[3];
rz(-0.04648195373854554) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0934844910545884) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0053430818898338615) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.007881066043860567) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1240407364824808) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.09016602162457703) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.014150681074671354) q[3];
cx q[2],q[3];
rx(-0.12025522571466549) q[0];
rz(0.0673813806015177) q[0];
rx(-0.01972566346102183) q[1];
rz(-0.16510904315679023) q[1];
rx(-0.03340739069790402) q[2];
rz(-0.09971849733157004) q[2];
rx(-0.1707923087142092) q[3];
rz(0.011465981644743353) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.11418534417477252) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(0.014324085796003876) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.021749206107600637) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.05285857927975369) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.07808799015404172) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.01882709789582073) q[3];
cx q[2],q[3];
rx(-0.1442884046633174) q[0];
rz(0.03124667908356455) q[0];
rx(-0.032950434618958004) q[1];
rz(-0.14715593848271685) q[1];
rx(0.019013076863770358) q[2];
rz(-0.05989168974512478) q[2];
rx(-0.12906989629375948) q[3];
rz(-0.006498380871802335) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.13031317285348612) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.027463949279778035) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.03399138748246587) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.04632825602177154) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.09450463220686342) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.05986983405548379) q[3];
cx q[2],q[3];
rx(-0.12429917461683783) q[0];
rz(-0.036436829920601065) q[0];
rx(-0.05973310482866063) q[1];
rz(-0.1573240991291986) q[1];
rx(-0.0064218302172234) q[2];
rz(-0.10810303083623252) q[2];
rx(-0.1443568744304555) q[3];
rz(-0.007087673258896803) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.1530039777745258) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.03848859999607514) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.03180123326911815) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.0980697714177045) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.14154006014197323) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.04159761441041455) q[3];
cx q[2],q[3];
rx(-0.10688959740849359) q[0];
rz(0.007635228787299627) q[0];
rx(-0.06332118334631451) q[1];
rz(-0.1674002445367163) q[1];
rx(0.03773909887384005) q[2];
rz(-0.11485685726524111) q[2];
rx(-0.17390276489211748) q[3];
rz(0.028491219549935862) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.12764202428450105) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.023821912388705787) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.03962764430003067) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.061508086986944654) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.11252574005521199) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.04271628455149027) q[3];
cx q[2],q[3];
rx(-0.20767983748354774) q[0];
rz(-0.04056730815507139) q[0];
rx(-0.08971709433122176) q[1];
rz(-0.16654686018395773) q[1];
rx(-0.010471686775075553) q[2];
rz(-0.08672966896415045) q[2];
rx(-0.1916113307499613) q[3];
rz(0.02903748357167367) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.09506495808185801) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.11103862611208487) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.011932805510943537) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.04034753979370173) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.06049349484769097) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.02964218790467923) q[3];
cx q[2],q[3];
rx(-0.1336935633814696) q[0];
rz(-0.07915188342974272) q[0];
rx(-0.013496083118377468) q[1];
rz(-0.12588337971340333) q[1];
rx(-0.03680226673380419) q[2];
rz(0.0013425215385079133) q[2];
rx(-0.12011945880263068) q[3];
rz(-0.03503444589406529) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.03790616631487112) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
h q[2];
sdg q[0];
h q[0];
sdg q[2];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.07412808484656733) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.038808117187751616) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1008737024366794) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
h q[3];
sdg q[1];
h q[1];
sdg q[3];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.09394146739977703) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.000755886568266274) q[3];
cx q[2],q[3];
rx(-0.2396872793145108) q[0];
rz(-0.11840699767706742) q[0];
rx(-0.05099331993028423) q[1];
rz(-0.12652844114427167) q[1];
rx(0.019472126075502347) q[2];
rz(-0.032507584265041814) q[2];
rx(-0.17618618882151624) q[3];
rz(-0.012145974478632512) q[3];