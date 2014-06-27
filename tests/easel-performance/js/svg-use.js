
var phet = phet || {};
phet.tests = phet.tests || {};

(function(){
  'use strict';

  var svgNS = 'http://www.w3.org/2000/svg';
  var xlinkNS = 'http://www.w3.org/1999/xlink';

  function buildSVG( main ) {
    var svg = document.createElementNS( svgNS, 'svg' );
    svg.style.position = 'absolute';
    svg.style.left = '0';
    svg.style.top = '0';
    svg.setAttribute( 'width', main.width() );
    svg.setAttribute( 'height', main.height() );
    main.append( svg );
    return svg;
  }

  function buildRect() {
    var rect = document.createElementNS( svgNS, 'rect' );

    rect.setAttribute( 'x', 0 );
    rect.setAttribute( 'y', 0 );
    rect.setAttribute( 'width', 20 );
    rect.setAttribute( 'height', 20 );
    rect.setAttribute( 'rx', 0 );
    rect.setAttribute( 'ry', 0 );

    rect.setAttribute( 'style', 'fill: #f00; stroke: #000;' );

    return rect;
  }

  function addUse( parent, id ) {
    var use = document.createElementNS( svgNS, 'use' );
    use.setAttributeNS( xlinkNS, 'xlink:href', '#' + id );
    parent.appendChild( use );
  }

  // groups (none, basic, complete) / simple (none, complete)

  var basicN = 200;

  phet.tests.svgUseSimpleNone = function( main ) {
    var svg = buildSVG( main );

    var rects = [];
    for ( var i = 0; i < basicN; i++ ) {
      var rect = buildRect();
      rects.push( rect );
      svg.appendChild( rect );
    }

    // return step function
    return function( timeElapsed ) {
      for ( var k = 0; k < basicN; k++ ) {
        rects[k].setAttribute( 'x', ( Math.random() - 0.5 ) * 400 + main.width() / 2 );
        rects[k].setAttribute( 'y', ( Math.random() - 0.5 ) * 400 + main.height() / 2 );
      }
      // group[0].transform.baseVal.getItem( 0 ).setMatrix( matrix.toSVGMatrix() );
    }
  };

  phet.tests.svgUseSimpleDuplicated = function( main ) {
    var svg = buildSVG( main );

    var defs = document.createElementNS( svgNS, 'defs' );
    svg.appendChild( defs );

    var uses = [];
    for ( var i = 0; i < basicN; i++ ) {
      var rect = buildRect();
      rect.setAttribute( 'id', i );
      defs.appendChild( rect );
      var use = document.createElementNS( svgNS, 'use' );
      use.setAttributeNS( xlinkNS, 'xlink:href', '#' + i );
      svg.appendChild( use );
      uses.push( use );
    }

    // return step function
    return function( timeElapsed ) {
      for ( var k = 0; k < basicN; k++ ) {
        uses[k].setAttribute( 'x', ( Math.random() - 0.5 ) * 400 + main.width() / 2 );
        uses[k].setAttribute( 'y', ( Math.random() - 0.5 ) * 400 + main.height() / 2 );
      }
      // group[0].transform.baseVal.getItem( 0 ).setMatrix( matrix.toSVGMatrix() );
    }
  };

  phet.tests.svgUseSimpleComplete = function( main ) {
    var svg = buildSVG( main );

    var defs = document.createElementNS( svgNS, 'defs' );
    svg.appendChild( defs );

    var rect = buildRect();
    rect.setAttribute( 'id', 'rr' );
    defs.appendChild( rect );

    var uses = [];
    for ( var i = 0; i < basicN; i++ ) {
      var use = document.createElementNS( svgNS, 'use' );
      use.setAttributeNS( xlinkNS, 'xlink:href', '#rr' );
      svg.appendChild( use );
      uses.push( use );
    }

    // return step function
    return function( timeElapsed ) {
      for ( var k = 0; k < basicN; k++ ) {
        uses[k].setAttribute( 'x', ( Math.random() - 0.5 ) * 400 + main.width() / 2 );
        uses[k].setAttribute( 'y', ( Math.random() - 0.5 ) * 400 + main.height() / 2 );
      }
      // group[0].transform.baseVal.getItem( 0 ).setMatrix( matrix.toSVGMatrix() );
    }
  };

  phet.tests.svgUseHeavyNone = function( main ) {
    var svg = buildSVG( main );

    var a = document.createElementNS( svgNS, 'g' );
    a.setAttribute( 'transform', dot.Matrix3.translation( main.width() / 2, main.height() / 2 ).getSVGTransform() );
    svg.appendChild( a );
    var b = document.createElementNS( svgNS, 'g' );
    a.appendChild( b );
    var c = document.createElementNS( svgNS, 'g' );
    a.appendChild( c );
    var d = document.createElementNS( svgNS, 'g' );
    d.setAttribute( 'transform', dot.Matrix3.translation( 100, 0 ).getSVGTransform() );
    b.appendChild( d );
    var e = document.createElementNS( svgNS, 'g' );
    c.appendChild( e );
    var f = document.createElementNS( svgNS, 'g' );
    d.appendChild( f );
    var g = document.createElementNS( svgNS, 'g' );
    e.appendChild( g );
    var h = document.createElementNS( svgNS, 'g' );
    f.appendChild( h );
    var i = document.createElementNS( svgNS, 'g' );
    b.appendChild( i );
    var j = document.createElementNS( svgNS, 'g' );
    b.appendChild( j );
    var k = document.createElementNS( svgNS, 'g' );
    b.appendChild( k );
    var l = document.createElementNS( svgNS, 'g' );
    c.appendChild( l );
    var m = document.createElementNS( svgNS, 'g' );
    l.appendChild( m );
    var n = document.createElementNS( svgNS, 'g' );
    m.appendChild( n );
    var o = document.createElementNS( svgNS, 'g' );
    n.appendChild( o );
    var p = document.createElementNS( svgNS, 'g' );
    o.appendChild( p );
    var q = document.createElementNS( svgNS, 'g' );
    p.appendChild( q );
    var r = document.createElementNS( svgNS, 'g' );
    q.appendChild( r );
    var s = document.createElementNS( svgNS, 'g' );
    r.appendChild( s );
    var t = document.createElementNS( svgNS, 'g' );
    s.appendChild( t );

    var u = buildRect();
    u.setAttribute( 'style', 'fill: #00f; stroke: #000;' );
    u.setAttribute( 'x', -200 );
    h.appendChild( u );

    var v = buildRect();
    v.setAttribute( 'y', 100 );
    b.appendChild( v );

    var w = buildRect();
    w.setAttribute( 'y', 150 );
    c.appendChild( w );

    var x = buildRect();
    t.appendChild( x );

    var y = buildRect();
    h.appendChild( y );

    var z = buildRect();
    z.setAttribute( 'style', 'fill: #0f0; stroke: #000;' );
    l.appendChild( z );

    var rotB = 0;
    var rotT = 0;

    // return step function
    return function( timeElapsed ) {
      b.setAttribute( 'transform', dot.Matrix3.rotation2( rotB += timeElapsed ).getSVGTransform() );
      t.setAttribute( 'transform', dot.Matrix3.rotation2( rotT += timeElapsed * 2 ).getSVGTransform() );
    }
  };

  phet.tests.svgUseHeavyComplete = function( main ) {
    var svg = buildSVG( main );

    var defs = document.createElementNS( svgNS, 'defs' );
    svg.appendChild( defs );

    var a = document.createElementNS( svgNS, 'g' );
    a.setAttribute( 'id', 'a' );
    a.setAttribute( 'transform', dot.Matrix3.translation( main.width() / 2, main.height() / 2 ).getSVGTransform() );
    defs.appendChild( a );
    var b = document.createElementNS( svgNS, 'g' );
    b.setAttribute( 'id', 'b' );
    defs.appendChild( b );
    addUse( a, 'b' );
    var c = document.createElementNS( svgNS, 'g' );
    c.setAttribute( 'id', 'c' );
    defs.appendChild( c );
    addUse( a, 'c' );
    var d = document.createElementNS( svgNS, 'g' );
    d.setAttribute( 'id', 'd' );
    d.setAttribute( 'transform', dot.Matrix3.translation( 100, 0 ).getSVGTransform() );
    defs.appendChild( d );
    addUse( b, 'd' );
    var e = document.createElementNS( svgNS, 'g' );
    e.setAttribute( 'id', 'e' );
    defs.appendChild( e );
    addUse( c, 'e' );
    var f = document.createElementNS( svgNS, 'g' );
    f.setAttribute( 'id', 'f' );
    defs.appendChild( f );
    addUse( d, 'f' );
    var g = document.createElementNS( svgNS, 'g' );
    g.setAttribute( 'id', 'g' );
    defs.appendChild( g );
    addUse( e, 'g' );
    var h = document.createElementNS( svgNS, 'g' );
    h.setAttribute( 'id', 'h' );
    defs.appendChild( h );
    addUse( f, 'h' );
    var i = document.createElementNS( svgNS, 'g' );
    i.setAttribute( 'id', 'i' );
    defs.appendChild( i );
    addUse( b, 'i' );
    var j = document.createElementNS( svgNS, 'g' );
    j.setAttribute( 'id', 'j' );
    defs.appendChild( j );
    addUse( b, 'j' );
    var k = document.createElementNS( svgNS, 'g' );
    k.setAttribute( 'id', 'k' );
    defs.appendChild( k );
    addUse( b, 'k' );
    var l = document.createElementNS( svgNS, 'g' );
    l.setAttribute( 'id', 'l' );
    defs.appendChild( l );
    addUse( c, 'l' );
    var m = document.createElementNS( svgNS, 'g' );
    m.setAttribute( 'id', 'm' );
    defs.appendChild( m );
    addUse( l, 'm' );
    var n = document.createElementNS( svgNS, 'g' );
    n.setAttribute( 'id', 'n' );
    defs.appendChild( n );
    addUse( m, 'n' );
    var o = document.createElementNS( svgNS, 'g' );
    o.setAttribute( 'id', 'o' );
    defs.appendChild( o );
    addUse( n, 'o' );
    var p = document.createElementNS( svgNS, 'g' );
    p.setAttribute( 'id', 'p' );
    defs.appendChild( p );
    addUse( o, 'p' );
    var q = document.createElementNS( svgNS, 'g' );
    q.setAttribute( 'id', 'q' );
    defs.appendChild( q );
    addUse( p, 'q' );
    var r = document.createElementNS( svgNS, 'g' );
    r.setAttribute( 'id', 'r' );
    defs.appendChild( r );
    addUse( q, 'r' );
    var s = document.createElementNS( svgNS, 'g' );
    s.setAttribute( 'id', 's' );
    defs.appendChild( s );
    addUse( r, 's' );
    var t = document.createElementNS( svgNS, 'g' );
    t.setAttribute( 'id', 't' );
    defs.appendChild( t );
    addUse( s, 't' );

    var u = buildRect();
    u.setAttribute( 'id', 'u' );
    u.setAttribute( 'style', 'fill: #00f; stroke: #000;' );
    u.setAttribute( 'x', -200 );
    defs.appendChild( u );
    addUse( h, 'u' );

    var v = buildRect();
    v.setAttribute( 'id', 'v' );
    v.setAttribute( 'y', 100 );
    defs.appendChild( v );
    addUse( b, 'v' );

    var w = buildRect();
    w.setAttribute( 'id', 'w' );
    w.setAttribute( 'y', 150 );
    defs.appendChild( w );
    addUse( c, 'w' );

    var x = buildRect();
    x.setAttribute( 'id', 'x' );
    defs.appendChild( x );
    addUse( t, 'x' );

    var y = buildRect();
    y.setAttribute( 'id', 'y' );
    defs.appendChild( y );
    addUse( h, 'y' );

    var z = buildRect();
    z.setAttribute( 'style', 'fill: #0f0; stroke: #000;' );
    z.setAttribute( 'id', 'z' );
    defs.appendChild( z );
    addUse( l, 'z' );

    addUse( svg, 'a' );

    var rotB = 0;
    var rotT = 0;

    // return step function
    return function( timeElapsed ) {
      b.setAttribute( 'transform', dot.Matrix3.rotation2( rotB += timeElapsed ).getSVGTransform() );
      t.setAttribute( 'transform', dot.Matrix3.rotation2( rotT += timeElapsed * 2 ).getSVGTransform() );
    }
  };


})();

