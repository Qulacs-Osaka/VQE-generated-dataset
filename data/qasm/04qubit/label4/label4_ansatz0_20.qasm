OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0749214076014205) q[2];
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
rz(-0.0457043006931114) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.056184924405908616) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.13524530508100654) q[3];
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
rz(0.006279792176470018) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.09600845652161173) q[3];
cx q[2],q[3];
rx(-0.10351106800297784) q[0];
rz(-0.054955003466533196) q[0];
rx(0.030335491566062533) q[1];
rz(-0.02892576380299085) q[1];
rx(-0.12102173312089268) q[2];
rz(-0.05251475533726902) q[2];
rx(-0.02090126263034393) q[3];
rz(-0.07565210130453622) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.09084690448301895) q[2];
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
rz(0.03929957032866732) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.04202897728815093) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.13545352639201613) q[3];
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
rz(0.009258505524338927) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.06460823141035305) q[3];
cx q[2],q[3];
rx(-0.11826270432300434) q[0];
rz(0.020842683501583192) q[0];
rx(-0.0271282865427042) q[1];
rz(-0.01582139854933642) q[1];
rx(-0.06172276949588421) q[2];
rz(-0.08460678502259535) q[2];
rx(-0.07776979734201825) q[3];
rz(-0.016450081817219735) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.07758761385445416) q[2];
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
rz(0.031241544160557126) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.04738597422191211) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.13918239106588817) q[3];
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
rz(0.03437777404074049) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.0036105450968219074) q[3];
cx q[2],q[3];
rx(-0.11317361632989044) q[0];
rz(-0.05821249724144925) q[0];
rx(-0.04415065503115211) q[1];
rz(-0.04217462104430599) q[1];
rx(-0.1154899221278342) q[2];
rz(-0.04197056192747873) q[2];
rx(-0.11807593791663844) q[3];
rz(-0.08517085977283957) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.11409167634352693) q[2];
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
rz(-0.035642258123409154) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.007682836442320788) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.0688106608349357) q[3];
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
rz(0.02863401356697031) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.05933480596225097) q[3];
cx q[2],q[3];
rx(-0.03751561224696165) q[0];
rz(0.014660364594454597) q[0];
rx(-0.06463354435319915) q[1];
rz(-0.08890793603934581) q[1];
rx(-0.10479731687972721) q[2];
rz(-0.047836650307517214) q[2];
rx(-0.10521534208812404) q[3];
rz(-0.03408028743845777) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.08342423059971807) q[2];
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
rz(-0.04789374889200172) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.04171891496370105) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.07736902968040081) q[3];
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
rz(-0.01353345545764502) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.01682541173750129) q[3];
cx q[2],q[3];
rx(-0.12869291153597867) q[0];
rz(-0.05193844526656309) q[0];
rx(-0.0074489904499970335) q[1];
rz(-0.13991781574552087) q[1];
rx(-0.10300139431830911) q[2];
rz(-0.0785701946583372) q[2];
rx(-0.08228608490752935) q[3];
rz(-0.05417589653715095) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.07544837657825453) q[2];
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
rz(-0.04457310260472941) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.04046945609346188) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.16023637740536853) q[3];
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
rz(0.038363984348215696) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.06319813637082476) q[3];
cx q[2],q[3];
rx(-0.057153382951260503) q[0];
rz(-0.03252427879271904) q[0];
rx(-0.020334905515203398) q[1];
rz(-0.08340838469924017) q[1];
rx(-0.07138881925894307) q[2];
rz(-0.05401776914700014) q[2];
rx(-0.17039380378373348) q[3];
rz(-0.06637687640462954) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.07578528930511248) q[2];
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
rz(0.041752649181336164) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.01743882970244303) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.07611261953833605) q[3];
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
rz(-0.04570916117861862) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.03741433289654402) q[3];
cx q[2],q[3];
rx(-0.08983116093257179) q[0];
rz(-0.04407455959090541) q[0];
rx(0.035020290782469946) q[1];
rz(-0.13117638101927362) q[1];
rx(-0.08234809028428425) q[2];
rz(-0.05879637686531715) q[2];
rx(-0.16526769234535235) q[3];
rz(-0.0026680529271930144) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.08793344244453485) q[2];
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
rz(0.0057795764015966195) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.020863854454499463) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.09569920452614047) q[3];
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
rz(-0.036680490060950754) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.014062050455347736) q[3];
cx q[2],q[3];
rx(-0.11780729502740465) q[0];
rz(-0.009644497462387533) q[0];
rx(0.05127274080461661) q[1];
rz(-0.08458322138034122) q[1];
rx(-0.04983230003123797) q[2];
rz(-0.10197423694369045) q[2];
rx(-0.09677392491358465) q[3];
rz(-0.08360268306779232) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.10511618942309439) q[2];
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
rz(-0.004256662834487046) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.03101147423925545) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.10441088924865768) q[3];
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
rz(0.01511034496620334) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.027567959595398348) q[3];
cx q[2],q[3];
rx(-0.0705598794595818) q[0];
rz(-0.014907045871864711) q[0];
rx(-0.020522485903758585) q[1];
rz(-0.09461036865869593) q[1];
rx(-0.012189670959816124) q[2];
rz(-0.08902877202313104) q[2];
rx(-0.12092998573361392) q[3];
rz(0.006561644782686425) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.14701203089329698) q[2];
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
rz(0.013046280624051478) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.03521596087531977) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1369661847618801) q[3];
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
rz(-0.009996953912903702) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.04126860279492823) q[3];
cx q[2],q[3];
rx(-0.09455187101493627) q[0];
rz(0.008971667537423018) q[0];
rx(0.021074551984811372) q[1];
rz(-0.09026546281116468) q[1];
rx(-0.07847873220124095) q[2];
rz(-0.042679328640982284) q[2];
rx(-0.12193728611675751) q[3];
rz(-0.05190770008927793) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.13374470240335448) q[2];
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
rz(0.02223554363489487) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.044245251127005704) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.11035571964051732) q[3];
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
rz(-0.003933686621629344) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.018423658217235638) q[3];
cx q[2],q[3];
rx(-0.09850185212503901) q[0];
rz(-0.05321260648047087) q[0];
rx(0.018262695747661567) q[1];
rz(-0.09435054310800305) q[1];
rx(-0.0664503577950401) q[2];
rz(-0.1249840595364673) q[2];
rx(-0.09703204483266858) q[3];
rz(-0.010554353099440856) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.09278090508235039) q[2];
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
rz(0.0303693666237837) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.10907591664195986) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1448960811682955) q[3];
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
rz(0.05444994988466136) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.006499248860621289) q[3];
cx q[2],q[3];
rx(-0.073434257556383) q[0];
rz(-0.05316230642500865) q[0];
rx(-0.015630083607589988) q[1];
rz(-0.08811658582390382) q[1];
rx(-0.010490200268941645) q[2];
rz(-0.0575351231224689) q[2];
rx(-0.10407220595725354) q[3];
rz(-0.03397805440435111) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.11437261147360465) q[2];
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
rz(-0.023425837161114575) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.06626283360646847) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.12462998873937417) q[3];
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
rz(0.000451307875195486) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.03431637017020028) q[3];
cx q[2],q[3];
rx(-0.07582915570405603) q[0];
rz(0.01864676356968182) q[0];
rx(-0.012795432264198704) q[1];
rz(-0.08831118842297081) q[1];
rx(-0.0181259781777459) q[2];
rz(-0.10442800054951575) q[2];
rx(-0.09602742860130282) q[3];
rz(0.0017500043865200242) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.10931114456263091) q[2];
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
rz(-0.001340654087436513) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.0691761479970547) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.14930978838053774) q[3];
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
rz(0.07467516916816375) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.006448009875945659) q[3];
cx q[2],q[3];
rx(-0.06533593116319919) q[0];
rz(-0.0574176011538776) q[0];
rx(0.012687938375277414) q[1];
rz(-0.061848958186908116) q[1];
rx(-0.045137612540548204) q[2];
rz(-0.11102410788930975) q[2];
rx(-0.10675565139606488) q[3];
rz(-0.017603927804342626) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.04674839333903282) q[2];
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
rz(-0.03862613721274069) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.05477899194551567) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.12049802881173152) q[3];
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
rz(0.054366177319629005) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.002115399427831485) q[3];
cx q[2],q[3];
rx(-0.1254951137954529) q[0];
rz(-0.05567944667774111) q[0];
rx(-0.057518983917382076) q[1];
rz(-0.12505276985488567) q[1];
rx(0.0215171286657102) q[2];
rz(-0.1372094218770302) q[2];
rx(-0.14072433127928188) q[3];
rz(0.03317222600386432) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.032267790071617036) q[2];
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
rz(-0.0176993949805176) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.027622244126186895) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.16395286124681213) q[3];
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
rz(-0.013516733720242431) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.00559592096540482) q[3];
cx q[2],q[3];
rx(-0.14668986751826518) q[0];
rz(-0.058314414991125545) q[0];
rx(0.0497629397143141) q[1];
rz(-0.03265819973195491) q[1];
rx(0.03713519950572742) q[2];
rz(-0.0939419113817973) q[2];
rx(-0.12822531408209764) q[3];
rz(0.03382170443020332) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.07114833844204184) q[2];
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
rz(-0.06042573575653462) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(-0.017012487141859968) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1949370264896339) q[3];
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
rz(0.038587570811939226) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.03766499750116107) q[3];
cx q[2],q[3];
rx(-0.19078672724600979) q[0];
rz(-0.005085373075789074) q[0];
rx(0.021665907638848722) q[1];
rz(-0.11003853019060936) q[1];
rx(0.03918306197868482) q[2];
rz(-0.058204813176179857) q[2];
rx(-0.2106666605549473) q[3];
rz(0.018204717628550804) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.13620706044473138) q[2];
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
rz(-0.1447134367139514) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.09237563553896896) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1666996624872936) q[3];
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
rz(-0.08172446867238048) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.027410334200995087) q[3];
cx q[2],q[3];
rx(-0.193947490033915) q[0];
rz(-0.04133763484087347) q[0];
rx(-0.020139544173171126) q[1];
rz(-0.12686754392730856) q[1];
rx(0.014251170164882025) q[2];
rz(-0.058547511874115526) q[2];
rx(-0.21059647930143813) q[3];
rz(0.057154859501790146) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.0895880651673864) q[2];
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
rz(-0.09339099887244387) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.07212360212017357) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.2226063845212506) q[3];
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
rz(-0.05635807703703349) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.008285019220369035) q[3];
cx q[2],q[3];
rx(-0.21374565157943345) q[0];
rz(-0.0779393179680585) q[0];
rx(0.0029766058248273455) q[1];
rz(-0.09145954906534565) q[1];
rx(-0.05570977751031683) q[2];
rz(-0.11442863038770194) q[2];
rx(-0.22432911110011908) q[3];
rz(0.008388430810003832) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.10376371457418933) q[2];
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
rz(-0.07510152044940399) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.03982618804803521) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.22154293806245176) q[3];
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
rz(-0.06825559479100321) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.052418521949526294) q[3];
cx q[2],q[3];
rx(-0.22213091391398412) q[0];
rz(-0.04630029233977622) q[0];
rx(-0.0066314880928653374) q[1];
rz(-0.04943938464489192) q[1];
rx(-0.0836793640533019) q[2];
rz(-0.10979393264155465) q[2];
rx(-0.14439476220136746) q[3];
rz(-0.04630412368895413) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.11874719323395566) q[2];
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
rz(-0.019556868968682452) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.08853301539732548) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1263334225856157) q[3];
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
rz(-0.1171910824733375) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(0.022203776673267037) q[3];
cx q[2],q[3];
rx(-0.1859640874808982) q[0];
rz(-0.004395393512941938) q[0];
rx(-0.01857988547808582) q[1];
rz(-0.09248790382703072) q[1];
rx(-0.052703049116292554) q[2];
rz(-0.08046918565902986) q[2];
rx(-0.239307810074143) q[3];
rz(-0.08250647051029901) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.15357717421879877) q[2];
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
rz(-0.014823859942770997) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.07642814791663179) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.1865338639893848) q[3];
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
rz(-0.1103672700643693) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.0244811992601763) q[3];
cx q[2],q[3];
rx(-0.23618245602321353) q[0];
rz(-0.030166577876318646) q[0];
rx(0.021486327354933433) q[1];
rz(-0.05428998549367597) q[1];
rx(0.003912729378294966) q[2];
rz(-0.09985700390756429) q[2];
rx(-0.21604220392082896) q[3];
rz(-0.04995656323281841) q[3];
h q[0];
h q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(-0.14229019040429453) q[2];
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
rz(0.01874320141604714) q[2];
cx q[1],q[2];
cx q[0],q[1];
h q[0];
s q[0];
h q[2];
s q[2];
cx q[0],q[1];
rz(0.06426761095629425) q[1];
cx q[0],q[1];
h q[1];
h q[3];
cx q[1],q[2];
cx q[2],q[3];
rz(-0.12651818627657405) q[3];
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
rz(-0.08073739995069036) q[3];
cx q[2],q[3];
cx q[1],q[2];
h q[1];
s q[1];
h q[3];
s q[3];
cx q[2],q[3];
rz(-0.011492791932662888) q[3];
cx q[2],q[3];
rx(-0.20792577977338764) q[0];
rz(-0.025589407169435748) q[0];
rx(-0.0025562592881477847) q[1];
rz(-0.08088883103908337) q[1];
rx(-0.06514515425270265) q[2];
rz(-0.10350313883718079) q[2];
rx(-0.2131092306242678) q[3];
rz(-0.0009210609732421572) q[3];