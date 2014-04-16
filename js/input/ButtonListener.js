// Copyright 2002-2013, University of Colorado

/**
 * Basic button handling.
 *
 * Uses 4 states:
 * up: mouse not over, not pressed
 * over: mouse over, not pressed
 * down: mouse over, pressed
 * out: mouse not over, pressed
 *
 * TODO: offscreen handling
 * TODO: fix enter/exit edge cases for moving nodes or add/remove child, and when touches are created
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/util/Trail' );
  var inherit = require( 'PHET_CORE/inherit' );
  
  var DownUpListener = require( 'SCENERY/input/DownUpListener' );
  
  /**
   * Options for the ButtonListener:
   *
   * mouseButton: 0
   * fireOnDown: false // default is to fire on 'up' after 'down', but passing fireOnDown: true will fire on 'down' instead
   * up: null          // Called on an 'up' state change, as up( event, oldState )
   * over: null        // Called on an 'over' state change, as over( event, oldState )
   * down: null        // Called on an 'down' state change, as down( event, oldState )
   * out: null         // Called on an 'out' state change, as out( event, oldState )
   * fire: null        // Called on a state change to/from 'down' (depending on fireOnDown), as fire( event ). Called after the triggering up/over/down event.
   */
  scenery.ButtonListener = function ButtonListener( options ) {

    this.buttonState = 'up'; // public: 'up', 'over', 'down' or 'out'
    
    this._overCount = 0; // how many pointers are over us (track a count, so we can handle multiple pointers gracefully)
    
    this._buttonOptions = options; // store the options object so we can call the callbacks
    
    var buttonListener = this;
    DownUpListener.call( this, {

      mouseButton: options.mouseButton || 0, // forward the mouse button, default to 0 (LMB)
      
      down: function( event, trail ) {
        event.pointer.active = true;
        buttonListener.setButtonState( event, 'down' );
      },
      
      up: function( event, trail ) {
        event.pointer.active = false;
        buttonListener.setButtonState( event, buttonListener._overCount > 0 ? 'over' : 'up' );
      }
    } );
  };

  var ButtonListener = scenery.ButtonListener;
  
  inherit( DownUpListener, ButtonListener, {

    setButtonState: function( event, state ) {
      if ( state !== this.buttonState ) {
        sceneryEventLog && sceneryEventLog( 'ButtonListener state change to ' + state + ' from ' + this.buttonState + ' for ' + ( this.downTrail ? this.downTrail.toString() : this.downTrail ) );
        var oldState = this.buttonState;
        
        this.buttonState = state;
        
        if ( this._buttonOptions[state] ) {
          this._buttonOptions[state]( event, oldState );
        }
        
        if ( this._buttonOptions.fire &&
             this._overCount > 0 &&
             ( this._buttonOptions.fireOnDown ? ( state === 'down' ) : ( oldState === 'down' ) ) ) {
          this._buttonOptions.fire( event );
        }
      }
    },
    
    enter: function( event ) {
      sceneryEventLog && sceneryEventLog( 'ButtonListener enter for ' + ( this.downTrail ? this.downTrail.toString() : this.downTrail ) );
      this._overCount++;
      if ( this._overCount === 1 ) {
        this.setButtonState( event, this.isDown ? 'down' : 'over' );
      }
    },

    exit: function( event ) {
      sceneryEventLog && sceneryEventLog( 'ButtonListener exit for ' + ( this.downTrail ? this.downTrail.toString() : this.downTrail ) );
      assert && assert( this._overCount > 0, 'Exit events not matched by an enter' );
      this._overCount--;
      if ( this._overCount === 0 ) {
        this.setButtonState( event, this.isDown ? 'out' : 'up' );
      }
    }
  } );

  //TODO delete this after work is completed on sun.Button and scenery.ButtonListener
  ButtonListener.TEST_LISTENER = new ButtonListener( {

    up: function( event, oldState ) {
      console.log( "ButtonListener.up oldState=" + oldState );
    },

    over: function( event, oldState ) {
      console.log( "ButtonListener.over oldState=" + oldState );
    },

    down: function( event, oldState ) {
      console.log( "ButtonListener.down oldState=" + oldState );
    },

    out: function( event, oldState ) {
      console.log( "ButtonListener.out oldState=" + oldState );
    },

    fire: function( event ) {
      console.log( "ButtonListener.fire" );
    }
  } );

  return ButtonListener;
} );


