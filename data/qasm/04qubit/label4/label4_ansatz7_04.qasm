OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-0.03739413769378164) q[0];
ry(-0.6049186552524874) q[1];
cx q[0],q[1];
ry(-0.5740329969527814) q[0];
ry(2.4355859456915643) q[1];
cx q[0],q[1];
ry(2.550260083528985) q[0];
ry(-1.281914010376997) q[2];
cx q[0],q[2];
ry(-1.556289446522365) q[0];
ry(-2.0816801493340265) q[2];
cx q[0],q[2];
ry(0.974953745894346) q[0];
ry(2.633640186262515) q[3];
cx q[0],q[3];
ry(2.730076379315697) q[0];
ry(-1.471929537253743) q[3];
cx q[0],q[3];
ry(1.1073997977283065) q[1];
ry(-2.893004543572086) q[2];
cx q[1],q[2];
ry(-1.0274416277451799) q[1];
ry(-0.973853330617039) q[2];
cx q[1],q[2];
ry(1.1625401238312696) q[1];
ry(-2.8273482631895264) q[3];
cx q[1],q[3];
ry(-0.5348923746717801) q[1];
ry(0.4699598280821631) q[3];
cx q[1],q[3];
ry(0.2604749488952837) q[2];
ry(1.4048509483136553) q[3];
cx q[2],q[3];
ry(1.454652202627411) q[2];
ry(-2.528953749414696) q[3];
cx q[2],q[3];
ry(0.3026820977562562) q[0];
ry(-2.192104310484148) q[1];
cx q[0],q[1];
ry(2.8917503899603325) q[0];
ry(2.266473692693316) q[1];
cx q[0],q[1];
ry(-2.3859390276100942) q[0];
ry(2.612728377529104) q[2];
cx q[0],q[2];
ry(-0.2727763668784575) q[0];
ry(1.6513661685127292) q[2];
cx q[0],q[2];
ry(1.4251382410362088) q[0];
ry(1.93843661744073) q[3];
cx q[0],q[3];
ry(-0.39574928264184006) q[0];
ry(-0.9400092579718837) q[3];
cx q[0],q[3];
ry(-3.131624300472465) q[1];
ry(0.4110915502388744) q[2];
cx q[1],q[2];
ry(1.5982868930554013) q[1];
ry(2.6114903291466267) q[2];
cx q[1],q[2];
ry(0.2732955549423117) q[1];
ry(-2.376274931158072) q[3];
cx q[1],q[3];
ry(1.5492860712296288) q[1];
ry(3.1395709681762733) q[3];
cx q[1],q[3];
ry(2.748843699194552) q[2];
ry(-0.5746038878291602) q[3];
cx q[2],q[3];
ry(-2.3869974309554847) q[2];
ry(0.6366165516839243) q[3];
cx q[2],q[3];
ry(1.7164920384022613) q[0];
ry(-0.5473899594968182) q[1];
cx q[0],q[1];
ry(2.4373074203345704) q[0];
ry(0.2638631898921837) q[1];
cx q[0],q[1];
ry(0.14997486646423397) q[0];
ry(1.2344368041982356) q[2];
cx q[0],q[2];
ry(-1.4846781399851938) q[0];
ry(1.3935223633178064) q[2];
cx q[0],q[2];
ry(-2.302473499948314) q[0];
ry(-1.3510574109307294) q[3];
cx q[0],q[3];
ry(0.48332587019599105) q[0];
ry(-2.8319084361556968) q[3];
cx q[0],q[3];
ry(-3.022835140533211) q[1];
ry(-0.8462187316120503) q[2];
cx q[1],q[2];
ry(-0.6691287449162253) q[1];
ry(1.7160509173189094) q[2];
cx q[1],q[2];
ry(3.0501496151583796) q[1];
ry(-2.449295915845229) q[3];
cx q[1],q[3];
ry(-1.5626450978390078) q[1];
ry(2.7334190841779105) q[3];
cx q[1],q[3];
ry(-0.6461803396704866) q[2];
ry(1.1032614308579016) q[3];
cx q[2],q[3];
ry(2.517776534364605) q[2];
ry(1.7954343285173824) q[3];
cx q[2],q[3];
ry(-0.46931691153231486) q[0];
ry(-1.3794784907163609) q[1];
cx q[0],q[1];
ry(-1.3216684693339982) q[0];
ry(3.086629097761051) q[1];
cx q[0],q[1];
ry(1.0707468634459871) q[0];
ry(-1.3034162360504182) q[2];
cx q[0],q[2];
ry(-2.722942834758868) q[0];
ry(2.1237320195905953) q[2];
cx q[0],q[2];
ry(0.5021423269822574) q[0];
ry(1.3067010844755103) q[3];
cx q[0],q[3];
ry(-3.0484639819454062) q[0];
ry(-0.7024866328304145) q[3];
cx q[0],q[3];
ry(1.6551959582753888) q[1];
ry(-1.740265717953383) q[2];
cx q[1],q[2];
ry(1.1921640616966778) q[1];
ry(2.9337995219668525) q[2];
cx q[1],q[2];
ry(0.05524645756387691) q[1];
ry(-2.5753014763666187) q[3];
cx q[1],q[3];
ry(3.0326595621617685) q[1];
ry(-1.7945302252379804) q[3];
cx q[1],q[3];
ry(-0.4244612514804773) q[2];
ry(0.5442300987496231) q[3];
cx q[2],q[3];
ry(-2.3699963980443117) q[2];
ry(-1.5484034463482432) q[3];
cx q[2],q[3];
ry(-1.056913785941625) q[0];
ry(-2.1204375528198938) q[1];
cx q[0],q[1];
ry(-1.1870607685818992) q[0];
ry(2.579326164465892) q[1];
cx q[0],q[1];
ry(0.1288703191971774) q[0];
ry(0.505918945348944) q[2];
cx q[0],q[2];
ry(1.970119005178506) q[0];
ry(-3.0512828645626198) q[2];
cx q[0],q[2];
ry(1.9894153997962754) q[0];
ry(1.1337237805884348) q[3];
cx q[0],q[3];
ry(0.9939243177892809) q[0];
ry(1.7281838619590395) q[3];
cx q[0],q[3];
ry(-2.3683359808234608) q[1];
ry(-2.8812326344667807) q[2];
cx q[1],q[2];
ry(1.101216940894251) q[1];
ry(-2.120829498363313) q[2];
cx q[1],q[2];
ry(-1.674672216706531) q[1];
ry(-2.17582366677537) q[3];
cx q[1],q[3];
ry(-2.6859535378554096) q[1];
ry(0.48715239905148344) q[3];
cx q[1],q[3];
ry(0.0048405464170412525) q[2];
ry(3.055297262049058) q[3];
cx q[2],q[3];
ry(2.295255696903572) q[2];
ry(2.4059384453880064) q[3];
cx q[2],q[3];
ry(0.6433056298123273) q[0];
ry(0.9516042801524565) q[1];
cx q[0],q[1];
ry(0.7831561570460774) q[0];
ry(2.0150037890377166) q[1];
cx q[0],q[1];
ry(0.7692342624708387) q[0];
ry(0.6200041581848054) q[2];
cx q[0],q[2];
ry(-0.027316125633723765) q[0];
ry(-3.094404278780086) q[2];
cx q[0],q[2];
ry(1.2450624363595315) q[0];
ry(2.9798746238351854) q[3];
cx q[0],q[3];
ry(-2.813670775644221) q[0];
ry(-1.6230698233546743) q[3];
cx q[0],q[3];
ry(2.6169362517988275) q[1];
ry(-0.08003832034495058) q[2];
cx q[1],q[2];
ry(-1.244002830481178) q[1];
ry(2.883431794310857) q[2];
cx q[1],q[2];
ry(0.9129662912697923) q[1];
ry(0.22854411939424069) q[3];
cx q[1],q[3];
ry(-1.2309232051789776) q[1];
ry(-1.7767064698927928) q[3];
cx q[1],q[3];
ry(-0.13434751022887959) q[2];
ry(2.978641313601867) q[3];
cx q[2],q[3];
ry(1.9889054375346729) q[2];
ry(-2.81162022470404) q[3];
cx q[2],q[3];
ry(2.2920915198211023) q[0];
ry(-1.7006265314828792) q[1];
cx q[0],q[1];
ry(0.9226037116088488) q[0];
ry(0.8945605879803057) q[1];
cx q[0],q[1];
ry(0.8525864413034875) q[0];
ry(-2.55384547195631) q[2];
cx q[0],q[2];
ry(-1.5502346276597814) q[0];
ry(1.0113903819620462) q[2];
cx q[0],q[2];
ry(1.0881653363852957) q[0];
ry(0.4337656813211952) q[3];
cx q[0],q[3];
ry(-2.5958265824360645) q[0];
ry(2.4930678604891785) q[3];
cx q[0],q[3];
ry(0.3702308147062414) q[1];
ry(0.13241844813964146) q[2];
cx q[1],q[2];
ry(-2.8688432454014383) q[1];
ry(-0.6142458900656769) q[2];
cx q[1],q[2];
ry(-0.21729043954689065) q[1];
ry(-1.6860436157296932) q[3];
cx q[1],q[3];
ry(-1.352986266216642) q[1];
ry(-2.797392837615521) q[3];
cx q[1],q[3];
ry(1.7257080063105406) q[2];
ry(1.8595883134027325) q[3];
cx q[2],q[3];
ry(-2.949402538563198) q[2];
ry(2.7787791949485032) q[3];
cx q[2],q[3];
ry(-2.8132533413685397) q[0];
ry(0.28398835091019314) q[1];
ry(-1.3063472826077895) q[2];
ry(0.5417218693756558) q[3];