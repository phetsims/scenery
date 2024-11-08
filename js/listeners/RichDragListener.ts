// Copyright 2024, University of Colorado Boulder

/**
 * A drag listener that supports both pointer and keyboard input. It is composed with a DragListener and a
 * KeyboardDragListener to support pointer input and alternative input. In the future it can support other
 * input modalities or PhET-specific features.
 *
 * Be sure to dispose of this listener when it is no longer needed.
 *
 * Options that are common to both listeners are provided directly to this listener. Options that are specific to
 * a particular listener can be provided through the dragListenerOptions or keyboardDragListenerOptions.
 *
 * Typical PhET usage will use a position Property in a model coordinate frame and look like this:
 *
 *     // A focusable Node that can be dragged with pointer or keyboard.
 *     const draggableNode = new Node( {
 *       tagName: 'div',
 *       focusable: true
 *     } );
 *
 *     const richDragListener = new RichDragListener( {
 *       positionProperty: someObject.positionProperty,
 *       transform: modelViewTransform
 *     } );
 *
 *     draggableNode.addInputListener( richDragListener );
 *
 * This listener works by implementing TInputListener and forwarding input events to the specific listeners. This is
 * how we support adding this listener through the scenery input listener API.
 *
 * @author Jesse Greenberg
 */

import DerivedProperty from '../../../axon/js/DerivedProperty.js';
import TReadOnlyProperty from '../../../axon/js/TReadOnlyProperty.js';
import optionize, { combineOptions } from '../../../phet-core/js/optionize.js';
import { AllDragListenerOptions, DragListener, DragListenerOptions, Hotkey, KeyboardDragListener, KeyboardDragListenerCallback, KeyboardDragListenerNullableCallback, KeyboardDragListenerOptions, PressedDragListener, PressListenerCallback, PressListenerDOMEvent, PressListenerEvent, PressListenerNullableCallback, scenery, SceneryEvent, TInputListener } from '../imports.js';
import { KeyboardDragListenerDOMEvent } from './KeyboardDragListener.js';

type SelfOptions = AllDragListenerOptions<DragListener | KeyboardDragListener, PressListenerDOMEvent | KeyboardDragListenerDOMEvent> & {

  // Additional options for the DragListener, OR any overrides for the DragListener that should
  // be used instead of AllDragListenerOptions. For example, if the DragListener should have different
  // mapPosition, you can provide that option here.
  dragListenerOptions?: DragListenerOptions;

  // Additional options for the KeyboardDragListener, OR any overrides for the KeyboardDragListener that should
  // be used instead of AllDragListenerOptions. For example, if the KeyboardDragListener should have different
  // mapPosition, you can provide that option here.
  keyboardDragListenerOptions?: KeyboardDragListenerOptions;
};

export type RichDragListenerOptions = SelfOptions;

export default class RichDragListener implements TInputListener {

  // The DragListener and KeyboardDragListener that are composed to create this listener. Public so that you can
  // add them to different Nodes if you need to, for cases where you need to access their properties directly,
  // or if you only need one of the listeners.
  public readonly dragListener: DragListener;
  public readonly keyboardDragListener: KeyboardDragListener;

  // True if the listener is currently pressed (DragListener OR KeyboardDragListener).
  public readonly isPressedProperty: TReadOnlyProperty<boolean>;

  // Implements TInputListener
  public readonly hotkeys: Hotkey[];

  public constructor( providedOptions?: RichDragListenerOptions ) {

    const options = optionize<RichDragListenerOptions>()( {

      // RichDragListenerOptions
      positionProperty: null,

      // Called when the drag is started, for any input type. If you want to determine the type of input, you can check
      // SceneryEvent.isFromPDOM or SceneryEvent.type. If you need a start behavior for a specific form of input,
      // provide a start callback for that listener's options. It will be called IN ADDITION to this callback.
      start: null,

      // Called when the drag is ended, for any input type. If you want to determine the type of input, you can check
      // SceneryEvent.isFromPDOM or SceneryEvent.type. If you need an end behavior for a specific form of input,
      // provide an end callback for that listener's options. It will be called IN ADDITION to this callback. The event
      // may be null for cases of interruption.
      end: null,

      // Called during the drag event, for any input type. If you want to determine the type of input, you can check
      // SceneryEvent.isFromPDOM or SceneryEvent.type. If you need a drag behavior for a specific form of input,
      // provide a drag callback for that listener's options. It will be called IN ADDITION to this callback.
      drag: null,
      transform: null,
      dragBoundsProperty: null,
      mapPosition: null,
      translateNode: false,
      dragListenerOptions: {},
      keyboardDragListenerOptions: {}
    }, providedOptions );

    // Options that will apply to both listeners.
    const sharedOptions = {
      positionProperty: options.positionProperty,
      transform: options.transform,
      dragBoundsProperty: options.dragBoundsProperty || undefined,
      mapPosition: options.mapPosition || undefined,
      translateNode: options.translateNode
    };

    //---------------------------------------------------------------------------------
    // Construct the DragListener and combine its options.
    //---------------------------------------------------------------------------------
    const wrappedDragListenerStart: PressListenerCallback<PressedDragListener> = ( event, listener ) => {

      // when the drag listener starts, interrupt the keyboard dragging
      this.keyboardDragListener.interrupt();

      options.start && options.start( event, listener );
      options.dragListenerOptions.start && options.dragListenerOptions.start( event, listener );
    };

    const wrappedDragListenerDrag: PressListenerCallback<PressedDragListener> = ( event, listener ) => {
      options.drag && options.drag( event, listener );
      options.dragListenerOptions.drag && options.dragListenerOptions.drag( event, listener );
    };

    const wrappedDragListenerEnd: PressListenerNullableCallback<PressedDragListener> = ( event, listener ) => {
      options.end && options.end( event, listener );
      options.dragListenerOptions.end && options.dragListenerOptions.end( event, listener );
    };

    const dragListenerOptions = combineOptions<DragListenerOptions<PressedDragListener>>(
      // target object
      {},
      // Options that apply to both, but can be overridden by provided listener-specific options
      sharedOptions,
      // Provided listener-specific options
      options.dragListenerOptions,
      // Options that cannot be overridden - see wrapped callbacks above
      {
        start: wrappedDragListenerStart,
        drag: wrappedDragListenerDrag,
        end: wrappedDragListenerEnd
      }
    );

    this.dragListener = new DragListener( dragListenerOptions );

    //---------------------------------------------------------------------------------
    // Construct the KeyboardDragListener and combine its options.
    //---------------------------------------------------------------------------------
    const wrappedKeyboardListenerStart: KeyboardDragListenerCallback = ( event, listener ) => {

      // when the drag listener starts, interrupt the pointer dragging
      this.dragListener.interrupt();

      options.start && options.start( event, listener );
      options.keyboardDragListenerOptions.start && options.keyboardDragListenerOptions.start( event, listener );
    };

    const wrappedKeyboardListenerDrag: KeyboardDragListenerCallback = ( event, listener ) => {
      options.drag && options.drag( event, listener );
      options.keyboardDragListenerOptions.drag && options.keyboardDragListenerOptions.drag( event, listener );
    };

    const wrappedKeyboardListenerEnd: KeyboardDragListenerNullableCallback = ( event, listener ) => {
      options.end && options.end( event, listener );
      options.keyboardDragListenerOptions.end && options.keyboardDragListenerOptions.end( event, listener );
    };

    const keyboardDragListenerOptions = combineOptions<KeyboardDragListenerOptions>(
      // target object
      {},
      // Options that apply to both, but can be overridden by provided listener-specific options
      sharedOptions,
      // Provided listener-specific options
      options.keyboardDragListenerOptions,
      // Options that cannot be overridden - see wrapped callbacks above
      {
        start: wrappedKeyboardListenerStart,
        drag: wrappedKeyboardListenerDrag,
        end: wrappedKeyboardListenerEnd
      }
    );

    this.keyboardDragListener = new KeyboardDragListener( keyboardDragListenerOptions );

    // The hotkeys from the keyboard listener are assigned to this listener so that they are activated for Nodes
    // where this listener is added.
    this.hotkeys = this.keyboardDragListener.hotkeys;

    this.isPressedProperty = DerivedProperty.or( [ this.dragListener.isPressedProperty, this.keyboardDragListener.isPressedProperty ] );
  }

  public get isPressed(): boolean {
    return this.dragListener.isPressed || this.keyboardDragListener.isPressed;
  }

  public dispose(): void {
    this.isPressedProperty.dispose();

    this.dragListener.dispose();
    this.keyboardDragListener.dispose();
  }

  /**
   * ********************************************************************
   * Forward input to both listeners
   * ********************************************************************
   */
  public interrupt(): void {
    this.dragListener.interrupt();
    this.keyboardDragListener.interrupt();
  }

  /**
   * ********************************************************************
   * Forward to the KeyboardListener
   * ********************************************************************
   */
  public keydown( event: SceneryEvent<KeyboardEvent> ): void {
    this.keyboardDragListener.keydown( event );
  }

  public focusout( event: SceneryEvent ): void {
    this.keyboardDragListener.focusout( event );
  }

  public focusin( event: SceneryEvent ): void {
    this.keyboardDragListener.focusin( event );
  }

  public cancel(): void {
    this.keyboardDragListener.cancel();
  }

  /**
   * ********************************************************************
   * Forward to the DragListener
   * ********************************************************************
   */
  public click( event: SceneryEvent<MouseEvent> ): void {
    this.dragListener.click( event );
  }

  public touchenter( event: PressListenerEvent ): void {
    this.dragListener.touchenter( event );
  }

  public touchmove( event: PressListenerEvent ): void {
    this.dragListener.touchmove( event );
  }

  public focus( event: SceneryEvent<FocusEvent> ): void {
    this.dragListener.focus( event );
  }

  public blur(): void {
    this.dragListener.blur();
  }

  public down( event: PressListenerEvent ): void {
    this.dragListener.down( event );
  }

  public up( event: PressListenerEvent ): void {
    this.dragListener.up( event );
  }

  public enter( event: PressListenerEvent ): void {
    this.dragListener.enter( event );
  }

  public move( event: PressListenerEvent ): void {
    this.dragListener.move( event );
  }

  public exit( event: PressListenerEvent ): void {
    this.dragListener.exit( event );
  }

  public pointerUp( event: PressListenerEvent ): void {
    this.dragListener.pointerUp( event );
  }

  public pointerCancel( event: PressListenerEvent ): void {
    this.dragListener.pointerCancel( event );
  }

  public pointerMove( event: PressListenerEvent ): void {
    this.dragListener.pointerMove( event );
  }

  public pointerInterrupt(): void {
    this.dragListener.pointerInterrupt();
  }
}

scenery.register( 'RichDragListener', RichDragListener );