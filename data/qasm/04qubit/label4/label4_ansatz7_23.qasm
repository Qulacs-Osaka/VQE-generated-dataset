OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
ry(-0.8370363613471399) q[0];
ry(1.2099192064498832) q[1];
cx q[0],q[1];
ry(-2.968442077191382) q[0];
ry(-2.103453346192382) q[1];
cx q[0],q[1];
ry(1.4047396890941144) q[0];
ry(-0.15844423370895255) q[2];
cx q[0],q[2];
ry(-1.7478767007535616) q[0];
ry(1.9961183761183419) q[2];
cx q[0],q[2];
ry(1.0562551175375865) q[0];
ry(2.687947370521518) q[3];
cx q[0],q[3];
ry(2.8328253651804634) q[0];
ry(-0.13499282808449384) q[3];
cx q[0],q[3];
ry(2.9759385888138232) q[1];
ry(2.4063703910246583) q[2];
cx q[1],q[2];
ry(0.35734224403504644) q[1];
ry(-3.0376348759707836) q[2];
cx q[1],q[2];
ry(0.9254659086173135) q[1];
ry(2.2100513279211027) q[3];
cx q[1],q[3];
ry(-1.4878862169795877) q[1];
ry(-0.0756064810761325) q[3];
cx q[1],q[3];
ry(-2.673160791086171) q[2];
ry(1.2586119494049481) q[3];
cx q[2],q[3];
ry(0.34952586272469527) q[2];
ry(2.7029574811675086) q[3];
cx q[2],q[3];
ry(1.2336252240813177) q[0];
ry(2.154922587902525) q[1];
cx q[0],q[1];
ry(-0.8340159250189862) q[0];
ry(-2.1599217605161494) q[1];
cx q[0],q[1];
ry(-2.9399792987595705) q[0];
ry(-2.9965197937826704) q[2];
cx q[0],q[2];
ry(1.5315460066407978) q[0];
ry(-1.9613288834931404) q[2];
cx q[0],q[2];
ry(-1.7659126942895433) q[0];
ry(0.6886985149878573) q[3];
cx q[0],q[3];
ry(1.6578635809261397) q[0];
ry(1.0577602402743431) q[3];
cx q[0],q[3];
ry(0.10522637380392018) q[1];
ry(-2.596643269448052) q[2];
cx q[1],q[2];
ry(-0.4679218735454249) q[1];
ry(1.8922932891785351) q[2];
cx q[1],q[2];
ry(0.12455531587169767) q[1];
ry(-1.6869890770095972) q[3];
cx q[1],q[3];
ry(-0.9569013588547984) q[1];
ry(-0.6474313874852289) q[3];
cx q[1],q[3];
ry(-2.464670080314778) q[2];
ry(2.4032261231631153) q[3];
cx q[2],q[3];
ry(1.6494370459082688) q[2];
ry(1.63551325284956) q[3];
cx q[2],q[3];
ry(1.9528373108682988) q[0];
ry(2.051754514536573) q[1];
cx q[0],q[1];
ry(-2.9849339214374573) q[0];
ry(-2.7288407028565342) q[1];
cx q[0],q[1];
ry(1.8855830577384058) q[0];
ry(-1.1227470684938794) q[2];
cx q[0],q[2];
ry(2.4430942734426844) q[0];
ry(2.766742120670231) q[2];
cx q[0],q[2];
ry(0.9907724890708849) q[0];
ry(-1.990315399238082) q[3];
cx q[0],q[3];
ry(-0.5899896746063665) q[0];
ry(-1.9593527072761985) q[3];
cx q[0],q[3];
ry(1.1835314627852314) q[1];
ry(1.5344833124936084) q[2];
cx q[1],q[2];
ry(2.761488404090365) q[1];
ry(-2.491769324883916) q[2];
cx q[1],q[2];
ry(-1.6504620713291691) q[1];
ry(1.99141780094055) q[3];
cx q[1],q[3];
ry(-0.7974042257319603) q[1];
ry(-1.3714810828065493) q[3];
cx q[1],q[3];
ry(-3.008812289306928) q[2];
ry(-1.171017658967906) q[3];
cx q[2],q[3];
ry(0.3596674983545576) q[2];
ry(-0.4137082202855824) q[3];
cx q[2],q[3];
ry(0.7654040615606137) q[0];
ry(0.2627538033017416) q[1];
cx q[0],q[1];
ry(1.4218097073819622) q[0];
ry(-1.234268569715165) q[1];
cx q[0],q[1];
ry(-1.65547159687536) q[0];
ry(-0.5972123955530763) q[2];
cx q[0],q[2];
ry(2.737951094243373) q[0];
ry(3.007562241216301) q[2];
cx q[0],q[2];
ry(-1.3855463129548233) q[0];
ry(-1.4036271337629094) q[3];
cx q[0],q[3];
ry(2.8418814233445366) q[0];
ry(-1.9750690506414994) q[3];
cx q[0],q[3];
ry(-1.2999484306935432) q[1];
ry(2.2263440547655957) q[2];
cx q[1],q[2];
ry(2.2107664404313976) q[1];
ry(2.939557856182079) q[2];
cx q[1],q[2];
ry(0.08615733670616764) q[1];
ry(1.3316951488212094) q[3];
cx q[1],q[3];
ry(-2.3426196252474467) q[1];
ry(2.9835567122270126) q[3];
cx q[1],q[3];
ry(2.4260822161155002) q[2];
ry(-0.3640151178720768) q[3];
cx q[2],q[3];
ry(-1.9094790417625003) q[2];
ry(-0.5039746865143906) q[3];
cx q[2],q[3];
ry(0.178347469134458) q[0];
ry(-0.47515409093987326) q[1];
cx q[0],q[1];
ry(1.9422801090989796) q[0];
ry(-2.5502526814254693) q[1];
cx q[0],q[1];
ry(-0.4161312690169625) q[0];
ry(2.6931326504153397) q[2];
cx q[0],q[2];
ry(0.6467625153103106) q[0];
ry(-1.3785198082107133) q[2];
cx q[0],q[2];
ry(0.961799791884836) q[0];
ry(-1.793878153209822) q[3];
cx q[0],q[3];
ry(-0.2451913707439104) q[0];
ry(-1.8847624821226763) q[3];
cx q[0],q[3];
ry(-2.9663586491123075) q[1];
ry(-0.022127313393679238) q[2];
cx q[1],q[2];
ry(-2.692340155720054) q[1];
ry(-0.3999879033568583) q[2];
cx q[1],q[2];
ry(-1.8807099512979482) q[1];
ry(0.35075545120269486) q[3];
cx q[1],q[3];
ry(0.7153639708965462) q[1];
ry(1.2177938120129013) q[3];
cx q[1],q[3];
ry(1.386441926060929) q[2];
ry(-2.1990245258096452) q[3];
cx q[2],q[3];
ry(-2.88616149589195) q[2];
ry(-2.0313432505480993) q[3];
cx q[2],q[3];
ry(0.06196421959797693) q[0];
ry(3.1032456564341997) q[1];
cx q[0],q[1];
ry(-0.4283419278843684) q[0];
ry(-0.9305579642987658) q[1];
cx q[0],q[1];
ry(-0.020409064041538194) q[0];
ry(-3.0853195297157288) q[2];
cx q[0],q[2];
ry(0.01718190419098775) q[0];
ry(-1.4049639425813023) q[2];
cx q[0],q[2];
ry(0.6567225506307497) q[0];
ry(1.2721635494037598) q[3];
cx q[0],q[3];
ry(-1.4030838959192347) q[0];
ry(1.3376615097882592) q[3];
cx q[0],q[3];
ry(-2.444920526975357) q[1];
ry(-0.7671313412493808) q[2];
cx q[1],q[2];
ry(-0.6155473453997544) q[1];
ry(-1.5721350115913022) q[2];
cx q[1],q[2];
ry(2.7339928452002793) q[1];
ry(-2.3213315880613283) q[3];
cx q[1],q[3];
ry(-1.967651251636712) q[1];
ry(-1.4745095536811998) q[3];
cx q[1],q[3];
ry(-1.1597144895667713) q[2];
ry(-1.6571341002418132) q[3];
cx q[2],q[3];
ry(-1.695852544034977) q[2];
ry(2.030166207385366) q[3];
cx q[2],q[3];
ry(-1.108231015085658) q[0];
ry(-2.328940332028289) q[1];
cx q[0],q[1];
ry(-0.45219749127828507) q[0];
ry(-2.1418632488985763) q[1];
cx q[0],q[1];
ry(2.200143207550411) q[0];
ry(2.073031624763755) q[2];
cx q[0],q[2];
ry(-2.9811832831983436) q[0];
ry(-2.6995175516104295) q[2];
cx q[0],q[2];
ry(2.6108900951203737) q[0];
ry(0.053611683578330904) q[3];
cx q[0],q[3];
ry(-0.36181571843025984) q[0];
ry(-0.6450852688017603) q[3];
cx q[0],q[3];
ry(-2.9801792850707365) q[1];
ry(1.21838294108099) q[2];
cx q[1],q[2];
ry(-1.4750651878254786) q[1];
ry(-1.1210382403968087) q[2];
cx q[1],q[2];
ry(-0.5740154529402819) q[1];
ry(-0.5404650349696875) q[3];
cx q[1],q[3];
ry(0.9366166772855555) q[1];
ry(-1.4882047017547004) q[3];
cx q[1],q[3];
ry(-1.2004281788980942) q[2];
ry(-0.9840468965370924) q[3];
cx q[2],q[3];
ry(-2.0750578872316443) q[2];
ry(-2.110828276014436) q[3];
cx q[2],q[3];
ry(2.618534461415369) q[0];
ry(0.37927891105937134) q[1];
cx q[0],q[1];
ry(2.806372231012355) q[0];
ry(0.14354737630297887) q[1];
cx q[0],q[1];
ry(-0.2321170642586372) q[0];
ry(1.2961880629484592) q[2];
cx q[0],q[2];
ry(1.9426858218215672) q[0];
ry(0.5042980723296866) q[2];
cx q[0],q[2];
ry(2.3986945448364163) q[0];
ry(-2.7298680428195317) q[3];
cx q[0],q[3];
ry(1.636889837383138) q[0];
ry(-2.8379151667815607) q[3];
cx q[0],q[3];
ry(-0.47340335343880197) q[1];
ry(-0.2740128335808185) q[2];
cx q[1],q[2];
ry(1.8789865981755662) q[1];
ry(-2.944007181842486) q[2];
cx q[1],q[2];
ry(1.0895845890757003) q[1];
ry(-1.625231764722581) q[3];
cx q[1],q[3];
ry(0.23360540290847442) q[1];
ry(0.3234342158288777) q[3];
cx q[1],q[3];
ry(-1.1018048222809984) q[2];
ry(2.9160614137145213) q[3];
cx q[2],q[3];
ry(1.7523814667959607) q[2];
ry(2.4018287087243277) q[3];
cx q[2],q[3];
ry(1.2716425547628762) q[0];
ry(-1.6611668337276926) q[1];
cx q[0],q[1];
ry(-2.490344447456093) q[0];
ry(3.0943928114682704) q[1];
cx q[0],q[1];
ry(-1.4724658539628517) q[0];
ry(-1.1871204164727045) q[2];
cx q[0],q[2];
ry(0.46792193471682564) q[0];
ry(2.3570684888924744) q[2];
cx q[0],q[2];
ry(0.7671731703668652) q[0];
ry(-0.7433711822192164) q[3];
cx q[0],q[3];
ry(-0.6899627976216668) q[0];
ry(0.17314283507794492) q[3];
cx q[0],q[3];
ry(-2.8170813058114783) q[1];
ry(-1.2575013366594412) q[2];
cx q[1],q[2];
ry(2.1801062817230727) q[1];
ry(-0.8327853482507429) q[2];
cx q[1],q[2];
ry(-1.9450669218611267) q[1];
ry(2.9465413377005825) q[3];
cx q[1],q[3];
ry(0.059980721660235936) q[1];
ry(-0.07045472593328791) q[3];
cx q[1],q[3];
ry(1.091329958139255) q[2];
ry(-3.0824135528499377) q[3];
cx q[2],q[3];
ry(1.3066000770137243) q[2];
ry(2.0574387808187398) q[3];
cx q[2],q[3];
ry(3.1134447854633107) q[0];
ry(-2.5183583694967435) q[1];
cx q[0],q[1];
ry(3.055807900162918) q[0];
ry(2.5279371055247797) q[1];
cx q[0],q[1];
ry(-2.5533387965990713) q[0];
ry(1.0399374862029471) q[2];
cx q[0],q[2];
ry(0.45157341062361755) q[0];
ry(-0.3016310569441312) q[2];
cx q[0],q[2];
ry(-2.178066052839748) q[0];
ry(-2.7265013873833954) q[3];
cx q[0],q[3];
ry(2.752429110547862) q[0];
ry(-0.6062322685653578) q[3];
cx q[0],q[3];
ry(-1.48713592222096) q[1];
ry(1.8959355914639593) q[2];
cx q[1],q[2];
ry(-2.9424679574789603) q[1];
ry(-2.092781042801167) q[2];
cx q[1],q[2];
ry(0.3763805536094106) q[1];
ry(2.7243611342321703) q[3];
cx q[1],q[3];
ry(0.6200092644766696) q[1];
ry(0.9137528277770344) q[3];
cx q[1],q[3];
ry(2.6953079073149486) q[2];
ry(1.3425887311534233) q[3];
cx q[2],q[3];
ry(-0.1067905543884855) q[2];
ry(1.031312128530856) q[3];
cx q[2],q[3];
ry(0.4096484808616516) q[0];
ry(-2.879412652813605) q[1];
cx q[0],q[1];
ry(-2.1267297955985938) q[0];
ry(-0.4098449617935558) q[1];
cx q[0],q[1];
ry(0.8894243687207729) q[0];
ry(-1.782356454970423) q[2];
cx q[0],q[2];
ry(-3.122143568674837) q[0];
ry(-1.7356041630891523) q[2];
cx q[0],q[2];
ry(-2.7431783654012163) q[0];
ry(0.30155512235048537) q[3];
cx q[0],q[3];
ry(-0.6596394124512588) q[0];
ry(2.9433742568860013) q[3];
cx q[0],q[3];
ry(-2.8366725870375094) q[1];
ry(-0.06786555135121031) q[2];
cx q[1],q[2];
ry(0.06543016077838804) q[1];
ry(-1.762042432421906) q[2];
cx q[1],q[2];
ry(-2.158699580471273) q[1];
ry(-2.199622378806452) q[3];
cx q[1],q[3];
ry(-2.8722276661562858) q[1];
ry(1.0919268065073053) q[3];
cx q[1],q[3];
ry(-1.087990724975184) q[2];
ry(0.19860525025348252) q[3];
cx q[2],q[3];
ry(1.27515859838517) q[2];
ry(-1.7623885329428295) q[3];
cx q[2],q[3];
ry(-2.0406608544454623) q[0];
ry(0.4367445901649445) q[1];
cx q[0],q[1];
ry(-0.40165819197213004) q[0];
ry(1.5201218585491414) q[1];
cx q[0],q[1];
ry(1.531699136289971) q[0];
ry(-2.873755253911037) q[2];
cx q[0],q[2];
ry(1.8162960762105227) q[0];
ry(1.054725783835412) q[2];
cx q[0],q[2];
ry(-0.806409035761642) q[0];
ry(2.1324386646147184) q[3];
cx q[0],q[3];
ry(0.7357798854173918) q[0];
ry(2.5089399423187544) q[3];
cx q[0],q[3];
ry(2.4993118951722924) q[1];
ry(-1.8319760354556165) q[2];
cx q[1],q[2];
ry(-1.796798387554995) q[1];
ry(-2.9208673627491537) q[2];
cx q[1],q[2];
ry(2.1582459156946134) q[1];
ry(-2.9353933316948955) q[3];
cx q[1],q[3];
ry(3.0347851418356666) q[1];
ry(1.411515205533716) q[3];
cx q[1],q[3];
ry(-0.23180135797023144) q[2];
ry(-0.040553957403790086) q[3];
cx q[2],q[3];
ry(2.71169590510406) q[2];
ry(-0.3293400315203012) q[3];
cx q[2],q[3];
ry(2.8866939356078123) q[0];
ry(0.4375127136363224) q[1];
cx q[0],q[1];
ry(0.21632741813714926) q[0];
ry(-1.323856089696977) q[1];
cx q[0],q[1];
ry(1.7830424628688284) q[0];
ry(2.4477632314142164) q[2];
cx q[0],q[2];
ry(-1.8552127291095462) q[0];
ry(-2.619003385806201) q[2];
cx q[0],q[2];
ry(0.7273015153131196) q[0];
ry(0.372123357065683) q[3];
cx q[0],q[3];
ry(1.6464696142721251) q[0];
ry(-0.33468242694698347) q[3];
cx q[0],q[3];
ry(2.8230107565506803) q[1];
ry(2.0639956940106923) q[2];
cx q[1],q[2];
ry(0.34423600000081395) q[1];
ry(2.267344318917021) q[2];
cx q[1],q[2];
ry(-2.48092192390567) q[1];
ry(-0.47537840117643704) q[3];
cx q[1],q[3];
ry(2.084176937127687) q[1];
ry(-1.3896984256811926) q[3];
cx q[1],q[3];
ry(-2.1601420223597296) q[2];
ry(-2.871549533394921) q[3];
cx q[2],q[3];
ry(0.10029463231327274) q[2];
ry(-0.19919797732739553) q[3];
cx q[2],q[3];
ry(0.3209519954856048) q[0];
ry(-2.272298181687033) q[1];
cx q[0],q[1];
ry(2.086325649886621) q[0];
ry(2.6238502049075834) q[1];
cx q[0],q[1];
ry(1.885335420174867) q[0];
ry(-1.6368921902129903) q[2];
cx q[0],q[2];
ry(1.757258653673523) q[0];
ry(-1.521317253281626) q[2];
cx q[0],q[2];
ry(1.2063647878613457) q[0];
ry(-2.0642942780839766) q[3];
cx q[0],q[3];
ry(2.4324629560235036) q[0];
ry(0.9973144713100037) q[3];
cx q[0],q[3];
ry(-1.170716709352578) q[1];
ry(2.7367258688746223) q[2];
cx q[1],q[2];
ry(-0.29542671059409464) q[1];
ry(1.666088978840034) q[2];
cx q[1],q[2];
ry(2.772874108620631) q[1];
ry(1.3366527036640083) q[3];
cx q[1],q[3];
ry(1.497639084276681) q[1];
ry(-3.091890604077724) q[3];
cx q[1],q[3];
ry(-0.491692742404954) q[2];
ry(0.7979070641828105) q[3];
cx q[2],q[3];
ry(0.9829973758339866) q[2];
ry(-0.567169311142786) q[3];
cx q[2],q[3];
ry(0.9031403391290249) q[0];
ry(-1.6461989449635992) q[1];
cx q[0],q[1];
ry(-0.3346630845117805) q[0];
ry(-1.091952048297607) q[1];
cx q[0],q[1];
ry(-2.917825106227408) q[0];
ry(1.5289025169439583) q[2];
cx q[0],q[2];
ry(-2.01311082305591) q[0];
ry(-3.017787720341811) q[2];
cx q[0],q[2];
ry(-2.2828081576727577) q[0];
ry(-2.5469497163421853) q[3];
cx q[0],q[3];
ry(-1.4512011202651967) q[0];
ry(1.8463811344393122) q[3];
cx q[0],q[3];
ry(-2.85582387934942) q[1];
ry(-1.6102163214194842) q[2];
cx q[1],q[2];
ry(1.6115343318135729) q[1];
ry(2.2258107065264596) q[2];
cx q[1],q[2];
ry(1.4191395423663247) q[1];
ry(0.9479810943884042) q[3];
cx q[1],q[3];
ry(2.138294965056469) q[1];
ry(-0.853168305816471) q[3];
cx q[1],q[3];
ry(0.291377840815656) q[2];
ry(0.9560652612229267) q[3];
cx q[2],q[3];
ry(1.4445791617743664) q[2];
ry(1.9197179994053633) q[3];
cx q[2],q[3];
ry(1.0304508856612653) q[0];
ry(-1.0234679010116408) q[1];
cx q[0],q[1];
ry(0.9673101592179227) q[0];
ry(2.3295822801476813) q[1];
cx q[0],q[1];
ry(-2.4331914438442497) q[0];
ry(2.7468930791344177) q[2];
cx q[0],q[2];
ry(-1.8704454635107552) q[0];
ry(3.0850621313649347) q[2];
cx q[0],q[2];
ry(2.2186276466538475) q[0];
ry(-1.9259223576677726) q[3];
cx q[0],q[3];
ry(-1.4977257402224362) q[0];
ry(0.7641197032752369) q[3];
cx q[0],q[3];
ry(3.1141715249871593) q[1];
ry(-1.600529189628726) q[2];
cx q[1],q[2];
ry(-2.2247557967169076) q[1];
ry(1.9070757400684801) q[2];
cx q[1],q[2];
ry(-0.8373602943935224) q[1];
ry(0.6331493135146805) q[3];
cx q[1],q[3];
ry(-2.8497353074389626) q[1];
ry(1.6569411224328974) q[3];
cx q[1],q[3];
ry(-0.2866078467621035) q[2];
ry(-1.4704362128088455) q[3];
cx q[2],q[3];
ry(-2.2950197310790976) q[2];
ry(1.3362834244769277) q[3];
cx q[2],q[3];
ry(-1.4444582449703187) q[0];
ry(-0.7952426973055123) q[1];
cx q[0],q[1];
ry(0.991732626597914) q[0];
ry(2.066913122596776) q[1];
cx q[0],q[1];
ry(-1.401630415251888) q[0];
ry(-0.4286926191136695) q[2];
cx q[0],q[2];
ry(-1.5836177475634108) q[0];
ry(-3.1076053850418215) q[2];
cx q[0],q[2];
ry(-1.1388414398832953) q[0];
ry(2.059630652329189) q[3];
cx q[0],q[3];
ry(-2.4429645617431346) q[0];
ry(-0.43205296160867146) q[3];
cx q[0],q[3];
ry(-1.8588470841025333) q[1];
ry(-1.8698449150217566) q[2];
cx q[1],q[2];
ry(0.8351091810804938) q[1];
ry(0.23231133711245489) q[2];
cx q[1],q[2];
ry(-0.8702377007535979) q[1];
ry(-2.327549170953524) q[3];
cx q[1],q[3];
ry(2.1719281438420133) q[1];
ry(2.842031655531799) q[3];
cx q[1],q[3];
ry(-2.0456909947615705) q[2];
ry(0.3904246469455953) q[3];
cx q[2],q[3];
ry(-1.7393250221182648) q[2];
ry(0.04015305837811223) q[3];
cx q[2],q[3];
ry(1.2044971545474126) q[0];
ry(0.3797312794157184) q[1];
cx q[0],q[1];
ry(2.0879421056449603) q[0];
ry(2.295195109495429) q[1];
cx q[0],q[1];
ry(3.1189481760290616) q[0];
ry(-1.1530611739348045) q[2];
cx q[0],q[2];
ry(3.04420003532125) q[0];
ry(2.8293352016360913) q[2];
cx q[0],q[2];
ry(-2.2206165197557173) q[0];
ry(-1.4956025079172086) q[3];
cx q[0],q[3];
ry(-0.623116358086758) q[0];
ry(-2.3773093878756364) q[3];
cx q[0],q[3];
ry(2.893996280324572) q[1];
ry(2.2666485968400183) q[2];
cx q[1],q[2];
ry(0.4170708177982115) q[1];
ry(-0.00587288951914446) q[2];
cx q[1],q[2];
ry(1.604755864756464) q[1];
ry(1.1069234003353905) q[3];
cx q[1],q[3];
ry(-2.80516613822167) q[1];
ry(-1.559099773631921) q[3];
cx q[1],q[3];
ry(-1.9643050984647235) q[2];
ry(1.3963663972914038) q[3];
cx q[2],q[3];
ry(-0.63952012256776) q[2];
ry(0.46151590543308973) q[3];
cx q[2],q[3];
ry(-1.7410768050802554) q[0];
ry(0.2147980979758609) q[1];
cx q[0],q[1];
ry(1.4134103923768315) q[0];
ry(1.37832232288278) q[1];
cx q[0],q[1];
ry(1.1572645247522078) q[0];
ry(-0.5298358916851953) q[2];
cx q[0],q[2];
ry(3.127793341659594) q[0];
ry(-1.4378047261303308) q[2];
cx q[0],q[2];
ry(-1.4760800652635657) q[0];
ry(-0.857578033977739) q[3];
cx q[0],q[3];
ry(-2.164619478130696) q[0];
ry(-0.030043294822005958) q[3];
cx q[0],q[3];
ry(0.9733637415261294) q[1];
ry(0.533445746722614) q[2];
cx q[1],q[2];
ry(1.0831146685157265) q[1];
ry(-2.0410018399037355) q[2];
cx q[1],q[2];
ry(-1.5512759574193766) q[1];
ry(-1.0971707211759338) q[3];
cx q[1],q[3];
ry(-0.24984302537715367) q[1];
ry(-2.77551259396339) q[3];
cx q[1],q[3];
ry(2.7388392375727166) q[2];
ry(-2.115189646079161) q[3];
cx q[2],q[3];
ry(1.5782009022117887) q[2];
ry(1.637675593628709) q[3];
cx q[2],q[3];
ry(-1.5644577387224523) q[0];
ry(-3.0405231952605116) q[1];
cx q[0],q[1];
ry(-0.8312221253236002) q[0];
ry(0.4084069916128298) q[1];
cx q[0],q[1];
ry(-1.390693693064776) q[0];
ry(2.278893087382693) q[2];
cx q[0],q[2];
ry(0.36921486295471073) q[0];
ry(0.908426252926449) q[2];
cx q[0],q[2];
ry(0.013085752554619686) q[0];
ry(1.5836039291417645) q[3];
cx q[0],q[3];
ry(1.7707098564482222) q[0];
ry(2.599135387835177) q[3];
cx q[0],q[3];
ry(0.09836284195978882) q[1];
ry(-0.15984526032819765) q[2];
cx q[1],q[2];
ry(2.4284546685189357) q[1];
ry(2.1970319059816426) q[2];
cx q[1],q[2];
ry(0.7044255105241186) q[1];
ry(1.2478298560146515) q[3];
cx q[1],q[3];
ry(-1.6174969166708646) q[1];
ry(-1.9895607759557938) q[3];
cx q[1],q[3];
ry(-0.1542182760866595) q[2];
ry(-2.029324070144485) q[3];
cx q[2],q[3];
ry(-0.23947717897714937) q[2];
ry(1.5600598081222197) q[3];
cx q[2],q[3];
ry(-2.7603731677483245) q[0];
ry(-1.4625336171892134) q[1];
cx q[0],q[1];
ry(-2.955781667867248) q[0];
ry(0.39059192298692036) q[1];
cx q[0],q[1];
ry(-2.256447668637237) q[0];
ry(1.5791757397764985) q[2];
cx q[0],q[2];
ry(-2.149536294763517) q[0];
ry(1.5358726142432944) q[2];
cx q[0],q[2];
ry(-2.988341371779216) q[0];
ry(-2.1894116436583673) q[3];
cx q[0],q[3];
ry(-0.3602462241256807) q[0];
ry(2.445402567564721) q[3];
cx q[0],q[3];
ry(2.940845540137314) q[1];
ry(0.17566181813179138) q[2];
cx q[1],q[2];
ry(-1.9994015957043052) q[1];
ry(2.4225672290443367) q[2];
cx q[1],q[2];
ry(-0.1346901741422926) q[1];
ry(0.07828397322241454) q[3];
cx q[1],q[3];
ry(-3.0137441191378525) q[1];
ry(-0.44033216852863344) q[3];
cx q[1],q[3];
ry(0.7855180375459807) q[2];
ry(-3.0957825705679998) q[3];
cx q[2],q[3];
ry(3.0585055562063777) q[2];
ry(-1.4652558232011683) q[3];
cx q[2],q[3];
ry(0.1174581341274159) q[0];
ry(3.096159051020584) q[1];
cx q[0],q[1];
ry(2.8644358223087005) q[0];
ry(2.037155126125941) q[1];
cx q[0],q[1];
ry(-0.8309386885546607) q[0];
ry(-1.384905326949048) q[2];
cx q[0],q[2];
ry(1.953149704505563) q[0];
ry(2.1902119637269735) q[2];
cx q[0],q[2];
ry(-0.32350344293819794) q[0];
ry(-1.279816953015974) q[3];
cx q[0],q[3];
ry(0.03555614630557713) q[0];
ry(-0.12965201648514646) q[3];
cx q[0],q[3];
ry(2.0159569899399) q[1];
ry(-2.183319276325728) q[2];
cx q[1],q[2];
ry(3.1159868558944526) q[1];
ry(-2.9813499198565334) q[2];
cx q[1],q[2];
ry(-0.6327221788109281) q[1];
ry(1.3310976070376688) q[3];
cx q[1],q[3];
ry(-1.4328641005832665) q[1];
ry(-2.8823544831012873) q[3];
cx q[1],q[3];
ry(-0.7155812326818877) q[2];
ry(0.33038251396829477) q[3];
cx q[2],q[3];
ry(-0.7176753661857973) q[2];
ry(1.2936021382395444) q[3];
cx q[2],q[3];
ry(-0.5559524344984318) q[0];
ry(1.253481342630112) q[1];
cx q[0],q[1];
ry(0.5163260380577229) q[0];
ry(0.47264320276205646) q[1];
cx q[0],q[1];
ry(-0.9736464416922539) q[0];
ry(0.991121930713032) q[2];
cx q[0],q[2];
ry(2.8310669185640407) q[0];
ry(-2.265346866416335) q[2];
cx q[0],q[2];
ry(-0.9575518772153657) q[0];
ry(1.6243213596904837) q[3];
cx q[0],q[3];
ry(1.5796521201844396) q[0];
ry(-1.828869966846497) q[3];
cx q[0],q[3];
ry(1.6848265006878584) q[1];
ry(-1.7853012376590491) q[2];
cx q[1],q[2];
ry(0.6112961844720006) q[1];
ry(-2.5914598482477382) q[2];
cx q[1],q[2];
ry(-1.0245724825541132) q[1];
ry(-0.6152994550335213) q[3];
cx q[1],q[3];
ry(1.814521832782801) q[1];
ry(0.18838621700026098) q[3];
cx q[1],q[3];
ry(-0.3438186102799703) q[2];
ry(-2.5888864507306026) q[3];
cx q[2],q[3];
ry(0.8195148184889764) q[2];
ry(2.757834245233626) q[3];
cx q[2],q[3];
ry(-1.6714386176324652) q[0];
ry(-1.8175802899108415) q[1];
cx q[0],q[1];
ry(1.6164048288049404) q[0];
ry(-2.596082070886117) q[1];
cx q[0],q[1];
ry(-0.5631688911132908) q[0];
ry(3.1288829987531157) q[2];
cx q[0],q[2];
ry(0.7584255258282273) q[0];
ry(2.0116149025951238) q[2];
cx q[0],q[2];
ry(-0.9294550744974179) q[0];
ry(2.0336456304017023) q[3];
cx q[0],q[3];
ry(-3.0643568335525626) q[0];
ry(2.337037115427827) q[3];
cx q[0],q[3];
ry(-0.14030829844794) q[1];
ry(2.4360121769265923) q[2];
cx q[1],q[2];
ry(0.6783772820167857) q[1];
ry(-1.2295143438907319) q[2];
cx q[1],q[2];
ry(0.5024906186594835) q[1];
ry(-2.7103745731357343) q[3];
cx q[1],q[3];
ry(-1.6419248212349995) q[1];
ry(2.2898216378973064) q[3];
cx q[1],q[3];
ry(0.4739135424972192) q[2];
ry(-2.6957906776768894) q[3];
cx q[2],q[3];
ry(2.5634337995462873) q[2];
ry(2.827510087377305) q[3];
cx q[2],q[3];
ry(2.6359556135563023) q[0];
ry(1.0286186843172391) q[1];
cx q[0],q[1];
ry(-3.048194366927237) q[0];
ry(-0.892302813418455) q[1];
cx q[0],q[1];
ry(-1.2698984556190496) q[0];
ry(0.2541919190568267) q[2];
cx q[0],q[2];
ry(1.9599318073708893) q[0];
ry(0.6728639648281849) q[2];
cx q[0],q[2];
ry(1.7291285771708012) q[0];
ry(0.7263896910458714) q[3];
cx q[0],q[3];
ry(-2.8530648950059505) q[0];
ry(1.380577044883115) q[3];
cx q[0],q[3];
ry(0.21275461237103382) q[1];
ry(1.548861772021228) q[2];
cx q[1],q[2];
ry(-1.8085873669214614) q[1];
ry(-1.8750300127015187) q[2];
cx q[1],q[2];
ry(-0.8717334583664718) q[1];
ry(-3.1347555192203416) q[3];
cx q[1],q[3];
ry(-0.48676662689654837) q[1];
ry(3.024388582205745) q[3];
cx q[1],q[3];
ry(0.9634583139006906) q[2];
ry(1.7717793735754837) q[3];
cx q[2],q[3];
ry(-0.8418405367957442) q[2];
ry(-2.0344481209933574) q[3];
cx q[2],q[3];
ry(2.5755627211789287) q[0];
ry(3.0272062922843492) q[1];
cx q[0],q[1];
ry(1.3313142183739037) q[0];
ry(1.9133222512161905) q[1];
cx q[0],q[1];
ry(2.3245110411263057) q[0];
ry(-1.7868351092575248) q[2];
cx q[0],q[2];
ry(-1.508450818836593) q[0];
ry(-0.7978648061551176) q[2];
cx q[0],q[2];
ry(-1.098460646156576) q[0];
ry(0.9795885153591896) q[3];
cx q[0],q[3];
ry(-0.09967910092649569) q[0];
ry(-1.3480232720585046) q[3];
cx q[0],q[3];
ry(-2.817840404584335) q[1];
ry(-1.632306868025771) q[2];
cx q[1],q[2];
ry(-0.4690233853730197) q[1];
ry(-2.3931845005094208) q[2];
cx q[1],q[2];
ry(-2.0401098474182398) q[1];
ry(-2.7200956043531064) q[3];
cx q[1],q[3];
ry(0.4576415920688639) q[1];
ry(1.7696969134365221) q[3];
cx q[1],q[3];
ry(1.0793621821129682) q[2];
ry(-0.9261064223567809) q[3];
cx q[2],q[3];
ry(0.39351615714747035) q[2];
ry(-2.863220525735084) q[3];
cx q[2],q[3];
ry(1.1892877181116042) q[0];
ry(-2.339292879465405) q[1];
ry(-0.5233837026285412) q[2];
ry(2.6001896026587845) q[3];