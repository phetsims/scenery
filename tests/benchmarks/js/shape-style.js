
var opts = {};

opts.A = 0;
opts.B = 1;
opts.C = 2;
opts.D = 3;
opts.E = 4;
opts.F = 5;
opts.G = 6;
opts.H = 7;
opts.I = 8;

opts.Super = function( type ) { this.type = type; };

opts.fA = function( args ) { opts.Super.call( this, opts.A ); this.args = args; this.a = args.a; }; opts.fA.prototype = Object.create( opts.Super.prototype ); opts.fA.prototype.constructor = opts.fA; opts.fA.prototype.op = function( context ) { context.add( this.a ); };
opts.fB = function( args ) { opts.Super.call( this, opts.B ); this.args = args; this.b = args.b; }; opts.fB.prototype = Object.create( opts.Super.prototype ); opts.fB.prototype.constructor = opts.fB; opts.fB.prototype.op = function( context ) { context.add( this.b ); };
opts.fC = function( args ) { opts.Super.call( this, opts.C ); this.args = args; this.c = args.c; }; opts.fC.prototype = Object.create( opts.Super.prototype ); opts.fC.prototype.constructor = opts.fC; opts.fC.prototype.op = function( context ) { context.add( this.c ); };
opts.fD = function( args ) { opts.Super.call( this, opts.D ); this.args = args; this.d = args.d; }; opts.fD.prototype = Object.create( opts.Super.prototype ); opts.fD.prototype.constructor = opts.fD; opts.fD.prototype.op = function( context ) { context.add( this.d ); };
opts.fE = function( args ) { opts.Super.call( this, opts.E ); this.args = args; this.e = args.e; }; opts.fE.prototype = Object.create( opts.Super.prototype ); opts.fE.prototype.constructor = opts.fE; opts.fE.prototype.op = function( context ) { context.add( this.e ); };
opts.fF = function( args ) { opts.Super.call( this, opts.F ); this.args = args; this.f = args.f; }; opts.fF.prototype = Object.create( opts.Super.prototype ); opts.fF.prototype.constructor = opts.fF; opts.fF.prototype.op = function( context ) { context.add( this.f ); };
opts.fG = function( args ) { opts.Super.call( this, opts.G ); this.args = args; this.g = args.g; }; opts.fG.prototype = Object.create( opts.Super.prototype ); opts.fG.prototype.constructor = opts.fG; opts.fG.prototype.op = function( context ) { context.add( this.g ); };
opts.fH = function( args ) { opts.Super.call( this, opts.H ); this.args = args; this.h = args.h; }; opts.fH.prototype = Object.create( opts.Super.prototype ); opts.fH.prototype.constructor = opts.fH; opts.fH.prototype.op = function( context ) { context.add( this.h ); };
opts.fI = function( args ) { opts.Super.call( this, opts.I ); this.args = args; this.i = args.i; }; opts.fI.prototype = Object.create( opts.Super.prototype ); opts.fI.prototype.constructor = opts.fI; opts.fI.prototype.op = function( context ) { context.add( this.i ); };

var Context = function() {
    this.n = 0;
};

Context.prototype = {
    constructor: Context,
    
    add: function( x ) {
        this.n += x;
    }
};

function switchyGeneral( ctx, x ) {
    switch( x.type ) {
        case opts.A: ctx.add( x.args.a ); break;
        case opts.B: ctx.add( x.args.b ); break;
        case opts.C: ctx.add( x.args.c ); break;
        case opts.D: ctx.add( x.args.d ); break;
        case opts.E: ctx.add( x.args.e ); break;
        case opts.F: ctx.add( x.args.f ); break;
        case opts.G: ctx.add( x.args.g ); break;
        case opts.H: ctx.add( x.args.h ); break;
        case opts.I: ctx.add( x.args.i ); break;
        default: throw new Error( 'what?!?' );
    }
}

function switchySpecific( ctx, x ) {
    switch( x.type ) {
        case opts.A: ctx.add( x.a ); break;
        case opts.B: ctx.add( x.b ); break;
        case opts.C: ctx.add( x.c ); break;
        case opts.D: ctx.add( x.d ); break;
        case opts.E: ctx.add( x.e ); break;
        case opts.F: ctx.add( x.f ); break;
        case opts.G: ctx.add( x.g ); break;
        case opts.H: ctx.add( x.h ); break;
        case opts.I: ctx.add( x.i ); break;
        default: throw new Error( 'what?!?' );
    }
}

var listOfThings = [];
for( var i = 0; i < 10; i++ ) {
    listOfThings.push( new opts.fA( 20 ) );
    listOfThings.push( new opts.fB( 40 ) );
    listOfThings.push( new opts.fC( 60 ) );
    listOfThings.push( new opts.fD( 80 ) );
    listOfThings.push( new opts.fE( 10 ) );
    listOfThings.push( new opts.fF( 30 ) );
    listOfThings.push( new opts.fG( 50 ) );
    listOfThings.push( new opts.fH( 70 ) );
    listOfThings.push( new opts.fI( 90 ) );
}
var count = listOfThings.length;


var ctx = new Context();
for( var i = 0; i < count; i++ ) {
    var x = listOfThings[i];
    switch( x.type ) {
        case opts.A: ctx.add( x.args.a ); break;
        case opts.B: ctx.add( x.args.b ); break;
        case opts.C: ctx.add( x.args.c ); break;
        case opts.D: ctx.add( x.args.d ); break;
        case opts.E: ctx.add( x.args.e ); break;
        case opts.F: ctx.add( x.args.f ); break;
        case opts.G: ctx.add( x.args.g ); break;
        case opts.H: ctx.add( x.args.h ); break;
        case opts.I: ctx.add( x.args.i ); break;
        default: throw new Error( 'what?!?' );
    }
}

var ctx = new Context();
for( var i = 0; i < count; i++ ) {
    var x = listOfThings[i];
    switch( x.type ) {
        case opts.A: ctx.add( x.a ); break;
        case opts.B: ctx.add( x.b ); break;
        case opts.C: ctx.add( x.c ); break;
        case opts.D: ctx.add( x.d ); break;
        case opts.E: ctx.add( x.e ); break;
        case opts.F: ctx.add( x.f ); break;
        case opts.G: ctx.add( x.g ); break;
        case opts.H: ctx.add( x.h ); break;
        case opts.I: ctx.add( x.i ); break;
        default: throw new Error( 'what?!?' );
    }
}

var ctx = new Context();
for( var i = 0; i < count; i++ ) {
    listOfThings[i].op( ctx );
}


