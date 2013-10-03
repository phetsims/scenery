// Copyright 2002-2013, University of Colorado

/**
 * An interval between two Trails. A trail being null means either 'from the start' or 'to the end', depending
 * on whether it is the first or second parameter to the constructor.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/util/Trail' );
  
  // start and end are of type {Trail} or null (indicates all the way to the start / end)
  scenery.RenderInterval = function RenderInterval( start, end ) {
    assert && assert( !start || !end || start.compare( end ) <= 0, 'RenderInterval parameters must not be out of order' );
    
    this.start = start;
    this.end = end;
  };
  var RenderInterval = scenery.RenderInterval;
  
  // assumes the intervals are disjoint, so we can just compare the start instance
  RenderInterval.compareDisjoint = function( x, y ) {
    // if they are both falsy, they should be the same
    if ( !x.start && !y.start ) { return 0; }
    
    // otherwise, since we are comparing the starts, null would signify 'before anything'
    if ( !x.start || !y.start ) { return x.start ? 1 : -1; }
    
    // otherwise our standard comparison
    return x.start.compare( y.start );
  };
  
  RenderInterval.prototype = {
    constructor: RenderInterval,
    
    reindex: function() {
      this.start && this.start.reindex();
      this.end && this.end.reindex();
    },
    
    isValidExclusive: function() {
      // like construction, but with strict inequality
      return !this.start || !this.end || this.start.compare( this.end ) < 0;
    },
    
    /*
     * Whether the union of this and the specified interval doesn't include any additional trails, when
     * both are treated as exclusive endpoints (exclusive between a and b). We also make the assumption
     * that a !== b || a === null for either interval, since otherwise it is not well defined.
     */
    exclusiveUnionable: function( interval ) {
      assert && assert ( this.isValidExclusive(), 'exclusiveUnionable requires exclusive intervals' );
      assert && assert ( interval.isValidExclusive(), 'exclusiveUnionable requires exclusive intervals' );
      return ( !this.start || !interval.end || this.start.compare( interval.end ) === -1 ) &&
             ( !this.end || !interval.start || this.end.compare( interval.start ) === 1 );
    },
    
    exclusiveContains: function( trail ) {
      assert && assert( trail );
      return ( !this.start || this.start.compare( trail ) < 0 ) && ( !this.end || this.end.compare( trail ) > 0 );
    },
    
    union: function( interval ) {
      // falsy checks since if a or b is null, we want that bound to be null
      var thisA = ( !this.start || ( interval.start && this.start.compare( interval.start ) === -1 ) );
      var thisB = ( !this.end || ( interval.end && this.end.compare( interval.end ) === 1 ) );
      
      return new RenderInterval(
        thisA ? this.start : interval.start,
        thisB ? this.end : interval.end
      );
    },
    
    toString: function() {
      return '[' + ( this.start ? this.start.toString() : this.start ) + ',' + ( this.end ? this.end.toString() : this.end ) + ']';
    }
  };
  
  return RenderInterval;
} );


