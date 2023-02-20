OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(1.4800979934407357) q[0];
rz(-1.8393277867778206) q[0];
ry(-2.394925834698632) q[1];
rz(0.3625011939453141) q[1];
ry(-0.10671221363743341) q[2];
rz(-1.7797810714582096) q[2];
ry(-2.813251186730257) q[3];
rz(1.6559954038839473) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.2022921354693068) q[0];
rz(1.9991299399942353) q[0];
ry(-0.22840787296762863) q[1];
rz(-2.5930795200978385) q[1];
ry(0.051057295699546756) q[2];
rz(2.2733449657165816) q[2];
ry(-1.1876008381350678) q[3];
rz(-2.1693443573827302) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.3041967378502386) q[0];
rz(1.1655322575627998) q[0];
ry(-0.5990811429575651) q[1];
rz(-1.6996316654620138) q[1];
ry(0.9953047554679534) q[2];
rz(2.2813480831082735) q[2];
ry(1.1263779672959702) q[3];
rz(0.4999114716213912) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-0.47448716470123187) q[0];
rz(2.8323794457161955) q[0];
ry(0.6677456629854028) q[1];
rz(0.06018015782795419) q[1];
ry(-0.9888050585700388) q[2];
rz(-1.2617952780239623) q[2];
ry(-2.8412407532992825) q[3];
rz(-1.9448394396408721) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.3464905733151946) q[0];
rz(-1.1700429707614193) q[0];
ry(-1.3587238247761366) q[1];
rz(-0.5719466735337297) q[1];
ry(0.24033186332882284) q[2];
rz(0.8607185706456382) q[2];
ry(1.6735027774511366) q[3];
rz(-1.8577419448286705) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.9067046815173712) q[0];
rz(-2.566518580953162) q[0];
ry(-1.4769669090452908) q[1];
rz(1.565324243768079) q[1];
ry(1.1811153770125962) q[2];
rz(2.691730836247742) q[2];
ry(-1.3885819119698477) q[3];
rz(-2.1104961826942668) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.0803373613751237) q[0];
rz(-2.9564614499848076) q[0];
ry(1.0668854833617232) q[1];
rz(-0.721521393481041) q[1];
ry(2.975160755765096) q[2];
rz(-0.24226211957769367) q[2];
ry(0.3741397539899376) q[3];
rz(-2.7851589932964482) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.4418296197174603) q[0];
rz(0.34873451254565424) q[0];
ry(1.0292191877320258) q[1];
rz(2.009798417122827) q[1];
ry(-1.4167578122743054) q[2];
rz(-1.4969469702770144) q[2];
ry(-1.3796634040483413) q[3];
rz(-0.37563782131540824) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.3544810600561084) q[0];
rz(-1.142735813661389) q[0];
ry(-2.522613679372793) q[1];
rz(-1.3054714788672113) q[1];
ry(-2.7472739884849795) q[2];
rz(2.7947502292249378) q[2];
ry(-1.427224745617521) q[3];
rz(-0.8106852878149793) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.394190983532675) q[0];
rz(-2.8772504801780783) q[0];
ry(-1.82244484082584) q[1];
rz(-2.38797258330331) q[1];
ry(-2.0029094692385296) q[2];
rz(-3.064021362267379) q[2];
ry(-2.7941991858633584) q[3];
rz(-2.6593199882457976) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.906236413421115) q[0];
rz(2.9385520149905657) q[0];
ry(-2.4810515333378365) q[1];
rz(-2.7803132675618665) q[1];
ry(3.116747348117234) q[2];
rz(-1.5869469823396205) q[2];
ry(1.5330552301969869) q[3];
rz(0.5275609251495359) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.04453734007201149) q[0];
rz(1.7223195560939164) q[0];
ry(-3.05457795715459) q[1];
rz(-0.4066221545494235) q[1];
ry(1.2954385419919752) q[2];
rz(2.2015192387192277) q[2];
ry(0.13073490216702227) q[3];
rz(-0.42718089026433326) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.9856082871611918) q[0];
rz(-1.7577362912145613) q[0];
ry(2.10495840996703) q[1];
rz(0.9399866920804729) q[1];
ry(0.6407740736264262) q[2];
rz(3.0094322170787233) q[2];
ry(-2.727822640029687) q[3];
rz(1.345780388010139) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.199555680547449) q[0];
rz(-1.7788588662487623) q[0];
ry(2.905786948917321) q[1];
rz(-1.3048172086653507) q[1];
ry(-1.6882497991960261) q[2];
rz(2.118620453310789) q[2];
ry(-2.403515043406465) q[3];
rz(0.7648498066318915) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-2.783771218210685) q[0];
rz(2.400803725171455) q[0];
ry(0.8709702388260354) q[1];
rz(1.394472522941057) q[1];
ry(-0.9568740780834615) q[2];
rz(-2.6352668644883726) q[2];
ry(2.09228067775779) q[3];
rz(-2.34898577663475) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.7145039051647675) q[0];
rz(0.8095217079949064) q[0];
ry(-2.6241382133323445) q[1];
rz(-1.3345907173465055) q[1];
ry(0.0318045844343473) q[2];
rz(0.12673939061680084) q[2];
ry(-2.9443530822528325) q[3];
rz(0.6788608637540224) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.4441655692530162) q[0];
rz(-2.0483330375186357) q[0];
ry(-0.10741912802547747) q[1];
rz(0.7484720469930988) q[1];
ry(2.9102359216601634) q[2];
rz(-0.8862946717605338) q[2];
ry(-1.7073153033697683) q[3];
rz(-0.5886018353318413) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.6661270828281034) q[0];
rz(0.8008687434339727) q[0];
ry(-1.0134623940701761) q[1];
rz(-0.4218760855851009) q[1];
ry(3.0498769222129294) q[2];
rz(-0.7084830238651579) q[2];
ry(2.4057880652142125) q[3];
rz(-1.263771607151864) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.11339138735418718) q[0];
rz(2.2787921713912507) q[0];
ry(-0.792478757582198) q[1];
rz(0.662886512472131) q[1];
ry(0.7529113594233899) q[2];
rz(0.8537775865075066) q[2];
ry(0.4727813086513706) q[3];
rz(1.6994183019340277) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.5853629429907306) q[0];
rz(-2.582165250454876) q[0];
ry(-2.8366485260849195) q[1];
rz(-2.1304241697904294) q[1];
ry(-0.9863206494384347) q[2];
rz(-0.21793324222645793) q[2];
ry(0.34049339621407443) q[3];
rz(2.8389337939088355) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.6979266859595303) q[0];
rz(1.2537997223134285) q[0];
ry(0.9869324678027295) q[1];
rz(1.050624911873986) q[1];
ry(-2.51127569556489) q[2];
rz(-1.3404050643640149) q[2];
ry(0.48093753404580497) q[3];
rz(0.07739854396398904) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.13572974788242928) q[0];
rz(1.5533813112101065) q[0];
ry(2.8772966291501874) q[1];
rz(2.2405524366430036) q[1];
ry(-2.306212118424606) q[2];
rz(2.0419565098818406) q[2];
ry(2.7509813769140736) q[3];
rz(0.1551517407901615) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.5141608997032882) q[0];
rz(-2.8703450149145975) q[0];
ry(1.0365831933153578) q[1];
rz(-1.8875737302507545) q[1];
ry(1.3082822804008423) q[2];
rz(-0.4060099454487496) q[2];
ry(-2.487099500318017) q[3];
rz(1.8366225677976593) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(2.197575407407251) q[0];
rz(0.3803315279933524) q[0];
ry(-3.052941980724041) q[1];
rz(0.5333237528892703) q[1];
ry(-0.36208787009335547) q[2];
rz(-2.4750601571484956) q[2];
ry(0.5860060604044568) q[3];
rz(-0.17551866483671663) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.7916981804432084) q[0];
rz(1.438818815822665) q[0];
ry(-1.4522234399401528) q[1];
rz(-0.5100489590035112) q[1];
ry(-1.3576150466681955) q[2];
rz(-0.5568627304533695) q[2];
ry(2.2194789554491976) q[3];
rz(-2.4274013576011004) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.2410258817662383) q[0];
rz(-2.565392481268425) q[0];
ry(-0.3378850564481557) q[1];
rz(-1.4037981674111633) q[1];
ry(-1.4572687242273876) q[2];
rz(1.4832884933896733) q[2];
ry(3.0890448197431795) q[3];
rz(-0.10145421675070754) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(1.8215285638488072) q[0];
rz(-2.5981124155816047) q[0];
ry(0.6160144765198139) q[1];
rz(-0.4784368374098627) q[1];
ry(-0.4766146946388533) q[2];
rz(0.24306340654320288) q[2];
ry(0.9103873151876604) q[3];
rz(-1.0223789657717575) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.9392912519287806) q[0];
rz(2.3743301216763744) q[0];
ry(0.5213788438854092) q[1];
rz(3.1277506663729224) q[1];
ry(0.06604045979721464) q[2];
rz(-0.8447417086328349) q[2];
ry(-0.36606761073392147) q[3];
rz(-1.663499703228344) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(0.5663676805822488) q[0];
rz(0.9837837268467098) q[0];
ry(-1.837973054964265) q[1];
rz(-2.9720944546151973) q[1];
ry(0.9748586878256509) q[2];
rz(1.4361993628120073) q[2];
ry(2.5259929701032102) q[3];
rz(-0.09744853093031212) q[3];
cz q[0],q[1];
cz q[2],q[3];
cz q[0],q[2];
cz q[1],q[3];
cz q[0],q[3];
cz q[1],q[2];
ry(-1.3950328084396855) q[0];
rz(2.664272032421486) q[0];
ry(0.5789522372576318) q[1];
rz(0.3364558566927131) q[1];
ry(0.13113801879991183) q[2];
rz(1.272337608586514) q[2];
ry(0.776425805132067) q[3];
rz(-0.295029016272351) q[3];