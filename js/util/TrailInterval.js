// Copyright 2002-2012, University of Colorado

/**
 * An interval between two Trails. A trail being null means either 'from the start' or 'to the end', depending
 * on whether it is the first or second parameter to the constructor.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/util/Trail' );
  
  // dataA and dataB are arbitrary types of data that can be attached, and are preserved on combination operations
  scenery.TrailInterval = function( a, b, dataA, dataB ) {
    assert && assert( !a || !b || a.compare( b ) <= 0, 'TrailInterval parameters must not be out of order' );
    
    // ensure that these trails will not be modified
    a && a.setImmutable();
    b && b.setImmutable();
    
    this.a = a;
    this.b = b;
    
    // data associated to each endpoint of the interval
    this.dataA = dataA;
    this.dataB = dataB;
  };
  var TrailInterval = scenery.TrailInterval;
  
  // assumes the intervals are disjoint, so we can just compare the starting (a) node
  TrailInterval.compareDisjoint = function( x, y ) {
    // if they are both falsy, they should be the same
    if ( !x.a && !y.a ) { return 0; }
    
    // otherwise, since we are comparing the starts, null would signify 'before anything'
    if ( !x.a || !y.a ) { return x.a ? 1 : -1; }
    
    // otherwise our standard comparison
    return x.a.compare( y.a );
  };
  
  TrailInterval.prototype = {
    constructor: TrailInterval,
    
    reindex: function() {
      this.a && this.a.reindex();
      this.b && this.b.reindex();
    },
    
    isValidExclusive: function() {
      // like construction, but with strict inequality
      return !this.a || !this.b || this.a.compare( this.b ) < 0;
    },
    
    /*
     * Whether the union of this and the specified interval doesn't include any additional trails, when
     * both are treated as exclusive endpoints (exclusive between a and b). We also make the assumption
     * that a !== b || a === null for either interval, since otherwise it is not well defined.
     */
    exclusiveUnionable: function( interval ) {
      assert && assert ( this.isValidExclusive(), 'exclusiveUnionable requires exclusive intervals' );
      assert && assert ( interval.isValidExclusive(), 'exclusiveUnionable requires exclusive intervals' );
      return ( !this.a || !interval.b || this.a.compare( interval.b ) === -1 ) &&
             ( !this.b || !interval.a || this.b.compare( interval.a ) === 1 );
    },
    
    exclusiveContains: function( trail ) {
      assert && assert( trail );
      return ( !this.a || this.a.compare( trail ) < 0 ) && ( !this.b || this.b.compare( trail ) > 0 );
    },
    
    union: function( interval ) {
      // falsy checks since if a or b is null, we want that bound to be null
      var thisA = ( !this.a || ( interval.a && this.a.compare( interval.a ) === -1 ) );
      var thisB = ( !this.b || ( interval.b && this.b.compare( interval.b ) === 1 ) );
      
      return new TrailInterval(
        thisA ? this.a : interval.a,
        thisB ? this.b : interval.b,
        thisA ? this.dataA : interval.dataA,
        thisB ? this.dataB : interval.dataB
      );
    },
    
    toString: function() {
      return '[' + ( this.a ? this.a.toString() : this.a ) + ',' + ( this.b ? this.b.toString() : this.b ) + ']';
    }
  };
  
  return TrailInterval;
} );


