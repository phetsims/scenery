// Copyright 2002-2012, University of Colorado

/**
 * A Canvas-style stateful (mutable) subpath, which tracks segments in addition to the points.
 *
 * See http://www.whatwg.org/specs/web-apps/current-work/multipage/the-canvas-element.html#concept-path
 * for the path / subpath Canvas concept.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  require( 'SCENERY/shapes/segments/Line' );
  
  scenery.Subpath = function() {
    this.points = [];
    this.segments = [];
    this.closed = false;
  };
  var Subpath = scenery.Subpath;
  Subpath.prototype = {
    addPoint: function( point ) {
      this.points.push( point );
    },
    
    addSegment: function( segment ) {
      if ( !segment.invalid ) {
        this.segments.push( segment );
      }
    },
    
    close: function() {
      this.closed = true;
    },
    
    getLength: function() {
      return this.points.length;
    },
    
    getFirstPoint: function() {
      return _.first( this.points );
    },
    
    getLastPoint: function() {
      return _.last( this.points );
    },
    
    getFirstSegment: function() {
      return _.first( this.segments );
    },
    
    getLastSegment: function() {
      return _.last( this.segments );
    },
    
    isDrawable: function() {
      return this.segments.length > 0;
    },
    
    isClosed: function() {
      return this.closed;
    },
    
    hasClosingSegment: function() {
      return !this.getFirstPoint().equals( this.getLastPoint() );
    },
    
    getClosingSegment: function() {
      assert && assert( this.hasClosingSegment(), 'Implicit closing segment unnecessary on a fully closed path' );
      return new scenery.Segment.Line( this.getLastPoint(), this.getFirstPoint() );
    }
  };
  
  return scenery.Subpath;
} );
