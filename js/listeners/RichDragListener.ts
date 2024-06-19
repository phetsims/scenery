// Copyright 2024, University of Colorado Boulder

/**
 * A drag listener that supports both pointer and keyboard input. It is composed with a DragListener and a
 * KeyboardDragListener to support pointer input, alternative input, sounds, and other PhET-specific features.
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

import TProperty from '../../../axon/js/TProperty.js';
import { DragListener, DragListenerOptions, Hotkey, KeyboardDragListener, KeyboardDragListenerOptions, PressedDragListener, PressListenerEvent, scenery, SceneryEvent, TInputListener } from '../imports.js';
import Vector2 from '../../../dot/js/Vector2.js';
import Transform3 from '../../../dot/js/Transform3.js';
import TReadOnlyProperty from '../../../axon/js/TReadOnlyProperty.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import optionize, { combineOptions } from '../../../phet-core/js/optionize.js';
import DerivedProperty from '../../../axon/js/DerivedProperty.js';

type SelfOptions = {

  // Called when the drag is started, for any input type. If you want to determine the type of input, you can check
  // SceneryEvent.isFromPDOM or SceneryEvent.type. If you need a start behavior for a specific form of input,
  // provide a start callback for that listener's options. It will be called IN ADDITION to this callback.
  start?: ( ( event: SceneryEvent, listener: DragListener | KeyboardDragListener ) => void ) | null;

  // Called during the drag event, for any input type. If you want to determine the type of input, you can check
  // SceneryEvent.isFromPDOM or SceneryEvent.type. If you need a drag behavior for a specific form of input,
  // provide a drag callback for that listener's options. It will be called IN ADDITION to this callback.
  drag?: ( ( event: SceneryEvent, listener: DragListener | KeyboardDragListener ) => void ) | null;

  // Called when the drag is ended, for any input type. If you want to determine the type of input, you can check
  // SceneryEvent.isFromPDOM or SceneryEvent.type. If you need an end behavior for a specific form of input,
  // provide an end callback for that listener's options. It will be called IN ADDITION to this callback. The event
  // may be null for cases of interruption.
  end?: ( ( event: SceneryEvent | null, listener: DragListener | KeyboardDragListener ) => void ) | null;

  // If provided, it will be synchronized with the drag position in the model coordinate frame. The optional transform
  // is applied.
  positionProperty?: Pick<TProperty<Vector2>, 'value'> | null;

  // If provided, this will be used to convert between the parent (view) and model coordinate frames. Most useful
  // when you also provide a positionProperty.
  transform?: Transform3 | TReadOnlyProperty<Transform3> | null;

  // If provided, the model position will be constrained to these bounds.
  dragBoundsProperty?: TReadOnlyProperty<Bounds2 | null> | null;

  // If provided, this allows custom mapping from the desired position (i.e. where the pointer is, or where the
  // RichKeyboardListener will set the position) to the actual position that will be used.
  mapPosition?: null | ( ( point: Vector2 ) => Vector2 );

  // If true, the target Node will be translated during the drag operation.
  translateNode?: boolean;

  // Additional options for the DragListener, OR any overrides for the DragListener that should
  // be used instead of the above options. For example, if the DragListener should have different
  // mapPosition, you can provide that option here.
  dragListenerOptions?: DragListenerOptions<DragListener>;

  // Additional options for the KeyboardDragListener, OR any overrides for the KeyboardDragListener that should
  // be used instead of the above options. For example, if the KeyboardDragListener should have different
  // mapPosition, you can provide that option here.
  keyboardDragListenerOptions?: KeyboardDragListenerOptions;
};

export type RichDragListenerOptions = SelfOptions;

export default class RichDragListener implements TInputListener {
  private readonly richPointerDragListener: DragListener;
  private readonly richKeyboardDragListener: KeyboardDragListener;

  // True if the listener is currently pressed (DragListener OR KeyboardDragListener).
  public readonly isPressedProperty: TReadOnlyProperty<boolean>;

  // Properties for each of the pressed states of the DragListener and KeyboardDragListener.
  public readonly keyboardListenerPressedProperty: TReadOnlyProperty<boolean>;
  public readonly pointerListenerPressedProperty: TReadOnlyProperty<boolean>;

  // Implements TInputListener
  public readonly hotkeys: Hotkey[];

  public constructor( providedOptions?: RichDragListenerOptions ) {

    const options = optionize<RichDragListenerOptions>()( {

      // RichDragListenerOptions
      positionProperty: null,
      start: null,
      end: null,
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
    const wrappedDragListenerStart = ( event: PressListenerEvent, listener: PressedDragListener ) => {

      // when the drag listener starts, interrupt the keyboard dragging
      this.richKeyboardDragListener.interrupt();

      options.start && options.start( event, listener );
      options.dragListenerOptions.start && options.dragListenerOptions.start( event, listener );
    };

    const wrappedDragListenerDrag = ( event: PressListenerEvent, listener: PressedDragListener ) => {
      options.drag && options.drag( event, listener );
      options.dragListenerOptions.drag && options.dragListenerOptions.drag( event, listener );
    };

    const wrappedDragListenerEnd = ( event: PressListenerEvent | null, listener: PressedDragListener ) => {
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

    this.richPointerDragListener = new DragListener( dragListenerOptions );

    //---------------------------------------------------------------------------------
    // Construct the KeyboardDragListener and combine its options.
    //---------------------------------------------------------------------------------
    const wrappedKeyboardListenerStart = ( event: SceneryEvent, listener: KeyboardDragListener ) => {

      // when the drag listener starts, interrupt the pointer dragging
      this.richPointerDragListener.interrupt();

      options.start && options.start( event, listener );
      options.keyboardDragListenerOptions.start && options.keyboardDragListenerOptions.start( event, listener );
    };

    const wrappedKeyboardListenerDrag = ( event: SceneryEvent, listener: KeyboardDragListener ) => {
      options.drag && options.drag( event, listener );
      options.keyboardDragListenerOptions.drag && options.keyboardDragListenerOptions.drag( event, listener );
    };

    const wrappedKeyboardListenerEnd = ( event: SceneryEvent | null, listener: KeyboardDragListener ) => {
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

    this.richKeyboardDragListener = new KeyboardDragListener( keyboardDragListenerOptions );

    // The hotkeys from the keyboard listener are assigned to this listener so that they are activated for Nodes
    // where this listener is added.
    this.hotkeys = this.richKeyboardDragListener.hotkeys;

    this.isPressedProperty = DerivedProperty.or( [ this.richPointerDragListener.isPressedProperty, this.richKeyboardDragListener.isPressedProperty ] );
    this.keyboardListenerPressedProperty = this.richKeyboardDragListener.isPressedProperty;
    this.pointerListenerPressedProperty = this.richPointerDragListener.isPressedProperty;
  }

  public get isPressed(): boolean {
    return this.richPointerDragListener.isPressed || this.richKeyboardDragListener.isPressed;
  }

  public dispose(): void {
    this.isPressedProperty.dispose();

    this.richPointerDragListener.dispose();
    this.richKeyboardDragListener.dispose();
  }

  /**
   * ********************************************************************
   * Forward input to both listeners
   * ********************************************************************
   */
  public interrupt(): void {
    this.richPointerDragListener.interrupt();
    this.richKeyboardDragListener.interrupt();
  }

  /**
   * ********************************************************************
   * Forward to the KeyboardListener
   * ********************************************************************
   */
  public keydown( event: SceneryEvent<KeyboardEvent> ): void {
    this.richKeyboardDragListener.keydown( event );
  }

  public focusout( event: SceneryEvent ): void {
    this.richKeyboardDragListener.focusout( event );
  }

  public focusin( event: SceneryEvent ): void {
    this.richKeyboardDragListener.focusin( event );
  }

  public cancel(): void {
    this.richKeyboardDragListener.cancel();
  }

  /**
   * ********************************************************************
   * Forward to the DragListener
   * ********************************************************************
   */
  public click( event: SceneryEvent<MouseEvent> ): void {
    this.richPointerDragListener.click( event );
  }

  public touchenter( event: PressListenerEvent ): void {
    this.richPointerDragListener.touchenter( event );
  }

  public touchmove( event: PressListenerEvent ): void {
    this.richPointerDragListener.touchmove( event );
  }

  public focus( event: SceneryEvent<FocusEvent> ): void {
    this.richPointerDragListener.focus( event );
  }

  public blur(): void {
    this.richPointerDragListener.blur();
  }

  public down( event: PressListenerEvent ): void {
    this.richPointerDragListener.down( event );
  }

  public up( event: PressListenerEvent ): void {
    this.richPointerDragListener.up( event );
  }

  public enter( event: PressListenerEvent ): void {
    this.richPointerDragListener.enter( event );
  }

  public move( event: PressListenerEvent ): void {
    this.richPointerDragListener.move( event );
  }

  public exit( event: PressListenerEvent ): void {
    this.richPointerDragListener.exit( event );
  }

  public pointerUp( event: PressListenerEvent ): void {
    this.richPointerDragListener.pointerUp( event );
  }

  public pointerCancel( event: PressListenerEvent ): void {
    this.richPointerDragListener.pointerCancel( event );
  }

  public pointerMove( event: PressListenerEvent ): void {
    this.richPointerDragListener.pointerMove( event );
  }

  public pointerInterrupt(): void {
    this.richPointerDragListener.pointerInterrupt();
  }
}

scenery.register( 'RichDragListener', RichDragListener );